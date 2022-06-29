#!/usr/bin/python

'''
Author: Marco Ragone, Computational Multiphase Transport Laboratory, University of Illinois at Chicago
'''

import numpy as np
import os

import random

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable






def save_epoch_results(training_results_folder_path,
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
                       colors):

    train_loss_learning_curve.append(train_loss)
    train_r2_learning_curve.append(r2_all_elements_train)

    np.save(os.path.join(training_results_folder_path, 'learning_curves/train_loss_learning_curve'),np.array(train_loss_learning_curve))
    np.save(os.path.join(training_results_folder_path, 'learning_curves/train_r2_learning_curve'),np.array(train_r2_learning_curve))

    test_loss_learning_curve.append(test_loss)
    test_r2_learning_curve.append(r2_all_elements_test)

    np.save(os.path.join(test_results_folder_path, 'learning_curves/test_loss_learning_curve'),np.array(test_loss_learning_curve))
    np.save(os.path.join(test_results_folder_path, 'learning_curves/test_r2_learning_curve'),np.array(test_r2_learning_curve))


    plot_learning_curves(np.array(train_loss_learning_curve),
                         np.array(test_loss_learning_curve),
                         np.array(train_r2_learning_curve),
                         np.array(test_r2_learning_curve),
                         chemical_symbols,
                         colors,
                         path = os.path.split(os.path.split(training_results_folder_path)[0])[0])

    if epoch % save_every == 0:
        model.save_weights(os.path.join(training_results_folder_path, 'weights/epoch-{}.h5'.format(epoch + 1)))

def plot_predictions(batch_images,
                       batch_labels,
                       batch_predictions,
                       chemical_symbols,
                       predictions_folder_path):

    cs = random.choice(list(np.arange(len(chemical_symbols)) + 1))

    for i in range(len(batch_images)):

        fig = plt.figure(figsize=(21,7))
        ax = fig.add_subplot(1, 3, 1)

        im = ax.imshow(batch_images[i,:,:,0], cmap='gray')
        plt.title('STEM Image',fontsize = 20)
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax = cax1)

        ax = fig.add_subplot(1, 3, 2)
        im = ax.imshow(batch_labels[i,:,:,cs - 1], cmap='jet')
        plt.title('{} Ground Truth'.format(chemical_symbols[cs - 1]), fontsize=20)
        divider = make_axes_locatable(ax)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax = cax2)

        ax = fig.add_subplot(1, 3, 3)
        im = ax.imshow(batch_predictions[i, :, :, cs - 1], cmap='jet')
        plt.title('{} Prediction'.format(chemical_symbols[cs - 1]), fontsize=20)
        divider = make_axes_locatable(ax)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax2)

        plt.tight_layout()

        fig.savefig(predictions_folder_path +'HEO_{}.png'.format(i + 1), bbox_inches='tight')

        plt.close(fig)

def plot_learning_curves(train_loss,
                         test_loss,
                         train_r2,
                         test_r2,
                         chemical_symbols,
                         colors,
                         path):

    epochs = np.arange(1,len(train_loss) + 1)

    fig = plt.figure(figsize=(20, 20))

    fig.add_subplot(2, 2, 1)
    plt.plot(epochs,train_loss,'bo-')
    plt.plot(epochs, test_loss, 'ro-')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(['Training','Test'])

    fig.add_subplot(2, 2, 2)
    plt.plot(epochs, train_r2[:,train_r2.shape[1] - 1], 'bo-')
    plt.plot(epochs, test_r2[:,test_r2.shape[1] - 1], 'ro-')
    plt.xlabel('Epochs')
    plt.ylabel('R2')
    plt.legend(['Training', 'Test'])

    fig.add_subplot(2, 2, 3)
    for i in range(train_r2.shape[1] - 1):
        plt.plot(epochs, train_r2[:, i], colors[i])

    plt.xlabel('Epochs')
    plt.ylabel('R2')
    plt.legend(chemical_symbols)

    fig.add_subplot(2, 2, 4)
    for i in range(test_r2.shape[1] - 1):
        plt.plot(epochs, test_r2[:, i], colors[i])

    plt.xlabel('Epochs')
    plt.ylabel('R2')
    plt.legend(chemical_symbols)

    plt.tight_layout()

    fig.savefig(os.path.join(path, 'learning_curves.png'), bbox_inches='tight')

    plt.close(fig)

