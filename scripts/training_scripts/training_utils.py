#!/usr/bin/python

'''
Author: Marco Ragone, Computational Multiphase Transport Laboratory, University of Illinois at Chicago
'''

import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
#import horovod.tensorflow as hvd

import logging
import platform

import sys
sys.path.append('../')

from config import*

from dataset_scripts.dataset_utils import*

from fcn_scripts.fcn import*
from fcn_scripts.model_utils import*

from results_scripts.R2_CHs_utils import*
from results_scripts.results_utils import*



def make_folder(path):
    if path and not os.path.exists(path):
        os.makedirs(path)
        print('Creating directory at {}'.format(path))

    else:

        print('Directory {} already exists!'.format(path))

def make_results_folder(path,train = True):

    make_folder(path)

    predictions_path = os.path.join(path, 'predictions/')
    make_folder(predictions_path)

    if train:

        weights_path = os.path.join(path, 'weights/')
        make_folder(weights_path)

    learning_curve_path = os.path.join(path, 'learning_curves/')
    make_folder(learning_curve_path)

def print_devices(num_devices):

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    logging.getLogger('tensorflow').setLevel(logging.FATAL)

    print("Running on host '{}'".format(platform.node()))

    if num_devices == 1:

        print('Running on {} device'.format(num_devices))

    else:

        print('Running on {} devices'.format(num_devices))


def compute_loss(labels, predictions,loss_object,batch_size):

    per_example_loss = loss_object(labels, predictions)

    per_example_loss /= tf.cast(tf.reduce_prod(tf.shape(labels)[1:]), tf.float32)

    return tf.nn.compute_average_loss(per_example_loss, global_batch_size = batch_size)


@tf.function
def train_step(batch,model,mp,loss_object,optimizer):

    batch_images, batch_labels = batch

    batch_size = len(batch_images)

    with tf.GradientTape() as tape:

        batch_predictions = model(batch_images, training=True)

        train_loss = compute_loss(batch_labels, batch_predictions, loss_object, batch_size)

        if mp:
            scaled_train_loss = optimizer.get_scaled_loss(train_loss)

    if mp:

        scaled_gradients = tape.gradient(scaled_train_loss, model.trainable_variables)

        gradients = optimizer.get_unscaled_gradients(scaled_gradients)

    else:

        gradients = tape.gradient(train_loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return train_loss, batch_predictions

@tf.function
def distributed_train_step(strategy,global_batch,model,mp,loss_object,optimizer):

    per_replica_losses, per_replica_predictions = strategy.run(train_step, args=(global_batch,model,mp,loss_object,optimizer,))

    loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,axis=None)

    predictions = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_predictions,axis=None)

    return loss,predictions


@tf.function
def train_step_hvd(hvd,batch,model,mp,loss_object,optimizer,first_batch):

    batch_images, batch_labels = batch

    with tf.GradientTape() as tape:

        batch_predictions = model(batch_images, training = True)

        train_loss = loss_object(batch_labels, batch_predictions)

        if mp:

            scaled_train_loss = optimizer.get_scaled_loss(train_loss)

    tape = hvd.DistributedGradientTape(tape)

    if mp:

        scaled_gradients = tape.gradient(scaled_train_loss, model.trainable_variables)

        gradients = optimizer.get_unscaled_gradients(scaled_gradients)

    else:

        gradients = tape.gradient(train_loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    if first_batch:

        hvd.broadcast_variables(model.variables, root_rank=0)

        hvd.broadcast_variables(optimizer.variables(), root_rank=0)

    return train_loss,batch_predictions


@tf.function
def test_step(batch,model,loss_object):

    batch_images, batch_labels = batch

    batch_size = len(batch_images)

    batch_predictions = model(batch_images, training = True)

    test_loss = compute_loss(batch_labels, batch_predictions, loss_object, batch_size)

    return test_loss, batch_predictions


@tf.function
def distributed_test_step(strategy,global_batch,model,loss_object):

    per_replica_losses, per_replica_predictions = strategy.run(test_step, args=(global_batch,model,loss_object,))

    loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    predictions = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_predictions, axis=None)

    return loss, predictions

@tf.function
def test_step_hvd(hvd,batch,model,loss_object,optimizer,first_batch):

    batch_images, batch_labels = batch

    batch_predictions = model(batch_images, training = True)

    test_loss = loss_object(batch_labels, batch_predictions)

    if first_batch:

        hvd.broadcast_variables(model.variables, root_rank = 0)

        hvd.broadcast_variables(optimizer.variables(), root_rank = 0)

    return test_loss,batch_predictions


def train_serial(config):

    results_folder_path = config.model_kwargs.results_folder_path
    data_folder_path = config.data_kwargs.data_folder_path
    batch_size = config.model_kwargs.batch_size
    mp = config.model_kwargs.mixed_precision
    num_chemical_elements = config.model_kwargs.num_chemical_elements
    first_epoch = config.model_kwargs.first_epoch
    num_epochs = config.model_kwargs.num_epochs
    save_every = config.model_kwargs.save_every
    chemical_symbols = list(np.array(list(config.plot_kwargs.color_elements.items()))[:, 0])
    colors = list(np.array(list(config.plot_kwargs.color_elements.items()))[:, 1])

    make_config_proto()

    print('Running serial model on 1 GPU')


    training_results_folder_path = os.path.join(results_folder_path,'training_results/')
    make_results_folder(training_results_folder_path)

    test_results_folder_path = os.path.join(results_folder_path,'test_results/')
    make_results_folder(test_results_folder_path, train = False)

    print('Training results will be saved at {}'.format(training_results_folder_path))
    print('Test results will be saved at {}'.format(test_results_folder_path))
    print('')

    training_data_folder_path = os.path.join(data_folder_path,'training_data/')
    test_data_folder_path = os.path.join(data_folder_path,'test_data/')

    train_dataset, num_batches_train = make_dataset(training_data_folder_path, batch_size)
    test_dataset, num_batches_test = make_dataset(test_data_folder_path, batch_size)


    model, optimizer, loss_object = compile_model(config)

    train_loss_learning_curve = []
    train_r2_learning_curve = []

    test_loss_learning_curve = []
    test_r2_learning_curve = []

    if first_epoch > 0:
        model.load_weights(os.path.join(training_results_folder_path, 'weights/epoch-{}.h5'.format(first_epoch)))

        train_loss_learning_curve = list(np.load(os.path.join(training_results_folder_path, 'learning_curve/train_loss_learning_curve.npy')))[:first_epoch + 1]
        train_r2_learning_curve = list(np.load(os.path.join(training_results_folder_path, 'learning_curve/train_r2_learning_curve.npy')))[:first_epoch + 1]

        test_loss_learning_curve = list(np.load(os.path.join(test_results_folder_path, 'learning_curve/test_loss_learning_curve.npy')))[:first_epoch + 1]
        test_r2_learning_curve = list(np.load(os.path.join(test_results_folder_path, 'learning_curve/test_r2_learning_curve.npy')))[:first_epoch + 1]

     
    print('Training starts!')
    for epoch in range(first_epoch, first_epoch + num_epochs):

        total_train_loss = 0.0

        total_average_r2_train = 0.0

        for train_batch_index, train_batch in enumerate(train_dataset):

            if  train_batch_index == 3:
                break

            train_images, train_labels = make_batch(train_batch,config)

            train_loss, train_predictions = train_step([train_images, train_labels], model, mp, loss_object, optimizer)

            total_train_loss = total_train_loss + train_loss

            processed_batches_train += 1

            train_loss = total_train_loss / (train_batch_index + 1)

            r2_CHs = R2_CHs(train_predictions, train_labels, num_chemical_elements)
            r2_all_elements_train = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_train += r2_all_elements_train[num_chemical_elements]
            r2_average_train = total_average_r2_train / (train_batch_index + 1)

            plot_predictions(train_images,
                             train_labels,
                             train_predictions,
                             chemical_symbols,
                             os.path.join(training_results_folder_path, 'predictions/'))

            if (train_batch_index + 1) % 1 == 0:
                print('Epoch [{}/{}] : Batch [{}/{}] : Train Loss = {:.4f}, Train R2 = {:.4f}'.format(epoch + 1,
                                                                                                      first_epoch + num_epochs,
                                                                                                      train_batch_index + 1,
                                                                                                      num_batches_train + 1,
                                                                                                      train_loss,
                                                                                                      r2_average_train))

        total_test_loss = 0

        total_average_r2_test = 0.0

        #for test_batch_index, test_batch in enumerate(test_dataset):
        for test_batch_index, test_batch in enumerate(train_dataset):

            if  test_batch_index == 3:
                break

            if  test_batch_index == 2000 // batch_size + 1:
                break

            test_images, test_labels = make_batch(test_batch,config)

            test_loss, test_predictions = test_step([test_images, test_labels], model, loss_object)

            plot_predictions(test_images,
                             test_labels,
                             test_predictions,
                             chemical_symbols,
                             os.path.join(test_results_folder_path, 'predictions/'))

            total_test_loss = total_test_loss + test_loss

            processed_batches_test += 1

            test_loss = total_test_loss / (test_batch_index + 1)

            r2_CHs = R2_CHs(test_predictions, test_labels, num_chemical_elements)
            r2_all_elements_test = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_test += r2_all_elements_test[num_chemical_elements]
            r2_average_test = total_average_r2_test / (test_batch_index + 1)

            if (test_batch_index + 1) % 1 == 0:
                print('Epoch [{}/{}] : Batch [{}/{}] : Test Loss = {:.4f}, Test R2 = {:.4f}'.format(epoch + 1,
                                                                                                   first_epoch + num_epochs,
                                                                                                   test_batch_index + 1,
                                                                                                   num_batches_test + 1,
                                                                                                   test_loss,
                                                                                                   r2_average_test))


        save_epoch_results(training_results_folder_path,
                           test_results_folder_path,
                           epoch,
                           save_every,
                           model,
                           train_loss_learning_curve,
                           train_r2_learning_curve,
                           test_loss_learning_curve,
                           test_r2_learning_curve,
                           train_loss,
                           r2_all_elements_train,
                           test_loss,
                           r2_all_elements_test,
                           chemical_symbols,
                           colors)


def train_data_parallel(config):

    results_folder_path = config.model_kwargs.results_folder_path
    data_folder_path = config.data_kwargs.data_folder_path
    batch_size = config.model_kwargs.batch_size
    mp = config.model_kwargs.mixed_precision
    num_chemical_elements = config.model_kwargs.num_chemical_elements
    first_epoch = config.model_kwargs.first_epoch
    num_epochs = config.model_kwargs.num_epochs
    save_every = config.model_kwargs.save_every
    chemical_symbols = list(np.array(list(config.plot_kwargs.color_elements.items()))[:, 0])
    colors = list(np.array(list(config.plot_kwargs.color_elements.items()))[:, 1])

    print('Running mirrored strategy model')
    print_devices(config.model_kwargs.n_gpus)

    make_config_proto()

    gpus_ids = []
    for i in range(config.model_kwargs.n_gpus):
        gpus_ids.append('/gpu:{}'.format(i))

    strategy = tf.distribute.MirroredStrategy(gpus_ids)
    print('Mirrored strategy is set on {} devices!'.format(len(gpus_ids)))

    training_results_folder_path = os.path.join(results_folder_path,'training_results/')
    make_results_folder(training_results_folder_path)

    test_results_folder_path = os.path.join(results_folder_path,'test_results/')
    make_results_folder(test_results_folder_path, train = False)

    print('Training results will be saved at {}'.format(training_results_folder_path))
    print('Test results will be saved at {}'.format(test_results_folder_path))
    print('')

    training_data_folder_path = os.path.join(data_folder_path, 'training_data/')
    test_data_folder_path = os.path.join(data_folder_path, 'test_data/')

    train_dataset, num_batches_train = make_dataset(training_data_folder_path, batch_size)
    test_dataset, num_batches_test = make_dataset(test_data_folder_path, batch_size)

    with strategy.scope():

        model, optimizer, loss_object = compile_model(config)

    train_loss_learning_curve = []
    train_r2_learning_curve = []

    test_loss_learning_curve = []
    test_r2_learning_curve = []

    if first_epoch > 0:

        model.load_weights(os.path.join(training_results_folder_path, 'weights/epoch-{}.h5'.format(first_epoch)))

        train_loss_learning_curve = list(np.load(os.path.join(training_results_folder_path, 'learning_curve/train_loss_learning_curve.npy')))[:first_epoch + 1]
        train_r2_learning_curve = list(np.load(os.path.join(training_results_folder_path, 'learning_curve/train_r2_learning_curve.npy')))[:first_epoch + 1]

        test_loss_learning_curve = list(np.load(os.path.join(test_results_folder_path, 'learning_curve/test_loss_learning_curve.npy')))[:first_epoch + 1]
        test_r2_learning_curve = list(np.load(os.path.join(test_results_folder_path, 'learning_curve/test_r2_learning_curve.npy')))[:first_epoch + 1]

    print('Training starts!')
    for epoch in range(first_epoch, first_epoch + num_epochs):


        total_train_loss = 0.0

        total_average_r2_train = 0.0

        for train_batch_index, train_batch in enumerate(train_dataset):

           # if train_batch_index == 3:
           #     break

            train_images, train_labels = make_batch(train_batch,config)

            train_loss, train_predictions = distributed_train_step(strategy,
                                                                   [train_images, train_labels],
                                                                   model,
                                                                   mp,
                                                                   loss_object,
                                                                   optimizer)

            total_train_loss = total_train_loss + train_loss

            train_loss = total_train_loss / (train_batch_index + 1)

            r2_CHs = R2_CHs(train_predictions, train_labels, num_chemical_elements)
            r2_all_elements_train = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_train += r2_all_elements_train[num_chemical_elements]
            r2_average_train = total_average_r2_train / (train_batch_index + 1)

            plot_predictions(train_images,
                             train_labels,
                             train_predictions,
                             chemical_symbols,
                             os.path.join(training_results_folder_path, 'predictions/'))

            if (train_batch_index + 1) % 1 == 0:
                print('Epoch [{}/{}] : Batch [{}/{}] : Train Loss = {:.4f}, Train R2 = {:.4f}'.format(epoch + 1,
                                                                                                      first_epoch + num_epochs,
                                                                                                      train_batch_index + 1,
                                                                                                      num_batches_train + 1,
                                                                                                      train_loss,
                                                                                                      r2_average_train))

        total_test_loss = 0

        total_average_r2_test = 0.0


        # for test_batch_index, test_batch in enumerate(test_dataset):
        for test_batch_index, test_batch in enumerate(train_dataset):

          #  if  test_batch_index == 3:
          #      break

            if test_batch_index == 2000 // batch_size + 1:
                break
            
             
            test_images, test_labels = make_batch(test_batch,config)

            test_loss, test_predictions = distributed_test_step(strategy,[test_images, test_labels], model, loss_object)

            plot_predictions(test_images,
                             test_labels,
                             test_predictions,
                             chemical_symbols,
                             os.path.join(test_results_folder_path, 'predictions/'))

            total_test_loss = total_test_loss + test_loss

            processed_batches_test += 1

            test_loss = total_test_loss / (test_batch_index + 1)

            r2_CHs = R2_CHs(test_predictions, test_labels, num_chemical_elements)
            r2_all_elements_test = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_test += r2_all_elements_test[num_chemical_elements]
            r2_average_test = total_average_r2_test / (test_batch_index + 1)

            if (test_batch_index + 1) % 1 == 0:
                print('Epoch [{}/{}] : Batch [{}/{}] : Test Loss = {:.4f}, Test R2 = {:.4f}'.format(epoch + 1,
                                                                                                   first_epoch + num_epochs,
                                                                                                   test_batch_index + 1,
                                                                                                   num_batches_test + 1,
                                                                                                   test_loss,
                                                                                                   r2_average_test))

        save_epoch_results(training_results_folder_path,
                           test_results_folder_path,
                           epoch,
                           save_every,
                           model,
                           train_loss_learning_curve,
                           train_r2_learning_curve,
                           test_loss_learning_curve,
                           test_r2_learning_curve,
                           train_loss,
                           r2_all_elements_train,
                           test_loss,
                           r2_all_elements_test,
                           chemical_symbols,
                           colors)


def train_model_parallel(config):

    results_folder_path = config.model_kwargs.results_folder_path
    data_folder_path = config.data_kwargs.data_folder_path
    batch_size = config.model_kwargs.batch_size
    mp = config.model_kwargs.mixed_precision
    num_chemical_elements = config.model_kwargs.num_chemical_elements
    first_epoch = config.model_kwargs.first_epoch
    num_epochs = config.model_kwargs.num_epochs
    save_every = config.model_kwargs.save_every
    chemical_symbols = list(np.array(list(config.plot_kwargs.color_elements.items()))[:,0])
    colors = list(np.array(list(config.plot_kwargs.color_elements.items()))[:, 1])

    make_config_proto_hvd(hvd)

    print('Running horovod model')
    print_devices(hvd.size())

    training_results_folder_path = os.path.join(results_folder_path,'training_results/')
    make_results_folder(training_results_folder_path)

    test_results_folder_path = os.path.join(results_folder_path,'test_results/')
    make_results_folder(test_results_folder_path, train = False)

    print('Training results will be saved at {}'.format(training_results_folder_path))
    print('Test results will be saved at {}'.format(test_results_folder_path))
    print('')

    training_data_folder_path = os.path.join(data_folder_path, 'training_data/')
    test_data_folder_path = os.path.join(data_folder_path, 'test_data/')

    train_dataset, num_batches_train = make_dataset(training_data_folder_path, batch_size)
   # test_dataset, num_batches_test = make_dataset(test_data_folder_path, batch_size)

    model, optimizer, loss_object = compile_model_hvd(config,hvd)

    train_loss_learning_curve = []
    train_r2_learning_curve = []

    test_loss_learning_curve = []
    test_r2_learning_curve = []

    if first_epoch > 0:

        model.load_weights(os.path.join(training_results_folder_path, 'weights/epoch-{}.h5'.format(first_epoch)))

        train_loss_learning_curve = list(np.load(os.path.join(training_results_folder_path, 'learning_curve/train_loss_learning_curve.npy')))[:first_epoch + 1]
        train_r2_learning_curve = list(np.load(os.path.join(training_results_folder_path, 'learning_curve/train_r2_learning_curve.npy')))[:first_epoch + 1]

        test_loss_learning_curve = list(np.load(os.path.join(test_results_folder_path, 'learning_curve/test_loss_learning_curve.npy')))[:first_epoch + 1]
        test_r2_learning_curve = list(np.load(os.path.join(test_results_folder_path, 'learning_curve/test_r2_learning_curve.npy')))[:first_epoch + 1]

    print('Training starts!')
    for epoch in range(first_epoch, first_epoch + num_epochs):

        total_train_loss = 0.0

        total_average_r2_train = 0.0


        for train_batch_index, train_batch in enumerate(train_dataset.take(10000 // hvd.size())):


            #if  train_batch_index == 3:
            #    break

            train_images, train_labels = make_batch(train_batch,config)

            train_loss, train_predictions = train_step_hvd(hvd,
                                                           [train_images, train_labels],
                                                           model,
                                                           mp,
                                                           loss_object,
                                                           optimizer,
                                                           train_batch_index == 0)

            total_train_loss = total_train_loss + train_loss

            processed_batches_train += 1

            train_loss = total_train_loss / (train_batch_index + 1)

            r2_CHs = R2_CHs(train_predictions, train_labels,num_chemical_elements)
            r2_all_elements_train = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_train += r2_all_elements_train[num_chemical_elements]
            r2_average_train = total_average_r2_train / (train_batch_index + 1)

            plot_predictions(train_images,
                             train_labels,
                             train_predictions,
                             chemical_symbols,
                             os.path.join(training_results_folder_path, 'predictions/'))

            if (train_batch_index + 1) % 1 == 0 and hvd.local_rank() == 0:

                print('Epoch [{}/{}] : Batch [{}/{}] : Train Loss = {:.4f}, Train R2 = {:.4f}'.format(epoch + 1,
                                                                                                      first_epoch + num_epochs,
                                                                                                      train_batch_index + 1,
                                                                                                      num_batches_train + 1,
                                                                                                      train_loss,
                                                                                                      r2_average_train))


        total_test_loss = 0

        total_average_r2_test = 0.0

        #for test_batch_index, test_batch in enumerate(test_dataset.take(10000 // hvd.size())):
        for test_batch_index, test_batch in enumerate(train_dataset.take(10000 // hvd.size())):

            #if  test_batch_index == 3:
            #    break

            num_batches_test = 2000 // batch_size
            if  test_batch_index == 2000 // batch_size + 1:
                break

            test_images, test_labels = make_batch(test_batch,config)

            test_loss, test_predictions =  test_step_hvd(hvd,
                                                         [test_images, test_labels],
                                                         model,
                                                         loss_object,
                                                         optimizer,
                                                         test_batch_index == 0)

            plot_predictions(test_images,
                               test_labels,
                               test_predictions,
                               chemical_symbols,
                               os.path.join(test_results_folder_path,'predictions/'))

            total_test_loss = total_test_loss + test_loss

            processed_batches_test += 1

            test_loss = total_test_loss / (test_batch_index + 1)

            r2_CHs = R2_CHs(test_predictions, test_labels,num_chemical_elements)
            r2_all_elements_test = r2_CHs.get_r2_all_elements_batch()

            total_average_r2_test += r2_all_elements_test[num_chemical_elements]
            r2_average_test = total_average_r2_test / (test_batch_index + 1)

            if hvd.local_rank() == 0:

                if (test_batch_index + 1) % 1 == 0:
                    print('Epoch [{}/{}] : Batch [{}/{}] : Test Loss = {:.4f}, Test R2 = {:.4f}'.format(epoch + 1,
                                                                                                       first_epoch + num_epochs,
                                                                                                       test_batch_index + 1,
                                                                                                       num_batches_test + 1,
                                                                                                       test_loss,
                                                                                                       r2_average_test))

        if hvd.local_rank() == 0:

            save_epoch_results(training_results_folder_path,
                               test_results_folder_path,
                               epoch,
                               save_every,
                               model,
                               train_loss_learning_curve,
                               train_r2_learning_curve,
                               test_loss_learning_curve,
                               test_r2_learning_curve,
                               train_loss,
                               r2_all_elements_train,
                               test_loss,
                               r2_all_elements_test,
                               chemical_symbols,
                               colors)



