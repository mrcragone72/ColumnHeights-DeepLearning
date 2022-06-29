#!/usr/bin/python

'''
Author: Marco Ragone, Computational Multiphase Transport Laboratory, University of Illinois at Chicago
'''

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision
#import horovod.tensorflow as hvd

import sys
sys.path.append('../')
from fcn_scripts.fcn import*


def make_model(FCN,input_shape, output_channels):

    input_channel = 1

    input_tensor = tf.keras.Input(shape = input_shape+(input_channel,))

    model = FCN(input_tensor, output_channels)

    return model


def make_pre_trained_model(pre_trained_model,weights_path,pre_trained_layers_at,freeze):

    print('Loading weights of pre-trained model from: {}'.format(weights_path))

    pre_trained_model.load_weights(weights_path)
    
    print('Removing score layer from pre-trained model')

    pre_trained_model = tf.keras.models.Model(pre_trained_model.input,
                              pre_trained_model.get_layer(pre_trained_layers_at).output)


    if freeze:
        for layer in pre_trained_model.layers:
            
            print('Freezing layer {}'.format(layer.name))
            layer.trainable = False
            
    return pre_trained_model

def make_top_model(pre_trained_model, input_shape,  output_channels):
    
    print('Adding score layer with {} channels to pre-trained model'.format( output_channels))
    
    input_channel = 1

    input_tensor = tf.keras.Input(shape = input_shape+(input_channel,))
    
    x = pre_trained_model(input_tensor)

    x = score_layer(x,channels = output_channels)

    top_model = tf.keras.Model(input_tensor,x)
    
    print('Top model is ready!')
    
    return top_model

def compile_model(config):

    if config.model_kwargs.pretrained:

        pre_trained_model = make_model(FCN,
                                       (config.model_kwargs.input_shape,) * 2,
                                       output_channels = config.model_kwargs.num_chemical_elements_pre_trained_model)

        pre_trained_model = make_pre_trained_model(pre_trained_model,
                                                   config.model_kwargs.pre_trained_weights_path,
                                                   config.model_kwargs.pre_trained_layers_at,
                                                   config.model_kwargs.freeze)

        model = make_top_model(pre_trained_model,
                               (config.model_kwargs.input_shape,) * 2,
                               config.model_kwargs.num_chemical_elements)

    else:

        model = make_model(FCN,
                           (config.model_kwargs.input_shape,) * 2,
                           output_channels=config.model_kwargs.num_chemical_elements)

    if config.model_kwargs.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=config.model_kwargs.learning_rate)

    if config.model_kwargs.loss == 'MSE':
        loss_object = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    if config.model_kwargs.mixed_precision:

        print('Running model with mixed precision')
        print('')

        policy = mixed_precision.Policy('mixed_float16')

        mixed_precision.set_policy(policy)

        optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale = 'dynamic')

    return model,optimizer,loss_object


def compile_model_hvd(config,hvd):

    if config.model_kwargs.pretrained:

        pre_trained_model = make_model(FCN,
                                       (config.model_kwargs.input_shape,) * 2,
                                       output_channels = config.model_kwargs.num_chemical_elements_pre_trained_model)

        pre_trained_model = make_pre_trained_model(pre_trained_model,
                                                   config.model_kwargs.pre_trained_weights_path,
                                                   config.model_kwargs.pre_trained_layers_at,
                                                   config.model_kwargs.freeze)

        model = make_top_model(pre_trained_model,
                               (config.model_kwargs.input_shape,) * 2,
                               config.model_kwargs.num_chemical_elements)

    else:

        model = make_model(FCN,
                           (config.model_kwargs.input_shape,) * 2,
                           output_channels=config.model_kwargs.num_chemical_elements)

    if config.model_kwargs.optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(lr=config.model_kwargs.learning_rate * hvd.size())

    if config.model_kwargs.loss == 'MSE':
        loss_object = tf.keras.losses.MeanSquaredError()

    if config.model_kwargs.mixed_precision:

        print('Running model with mixed precision')
        print('')

        policy = mixed_precision.Policy('mixed_float16')

        mixed_precision.set_policy(policy)

        optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale = 'dynamic')

    return model,optimizer,loss_object

