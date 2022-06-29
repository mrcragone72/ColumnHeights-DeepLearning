#!/usr/bin/python

'''
Author: Marco Ragone, Computational Multiphase Transport Laboratory, University of Illinois at Chicago
'''

import numpy as np
import os
from skimage.filters import gaussian

import tensorflow as tf




class Random_Imaging():

    """
    * Random_Imaging: class to perform random transformations on the images.
    The random transformations include random blur, random brightness, random contrast,
    random gamma and random flipping/rotation. Input parameters:

    - image: image to which apply the random transformations.
    - labels: included for the random flipping/rotation.

    """

    def __init__ (self,image,labels):

        self.image = image
        self.labels = labels

    def random_blur(self,image,low,high):

        sigma = np.random.uniform(low,high)

        image = gaussian(image,sigma)

        return image

    def random_brightness(self,image, low, high, rnd=np.random.uniform):

        image = image+rnd(low,high)

        return image

    def random_contrast(self,image, low, high, rnd=np.random.uniform):

        mean = np.mean(image)

        image = (image-mean)*rnd(low,high)+mean

        return image

    def random_gamma(self,image, low, high, rnd=np.random.uniform):

        min = np.min(image)

        image = (image-min)*rnd(low,high)+min

        return image

    def random_flip(self,image, labels, rnd=np.random.rand()):

        if rnd < 0.5:

            image[0,:,:,:] = np.fliplr(image[0,:,:,:])

            labels[0,:,:,:] = np.fliplr(labels[0,:,:,:])

        if rnd > 0.5:

            image[0,:,:,:] = np.flipud(image[0,:,:,:])

            labels[0,:,:,:] = np.flipud(labels[0,:,:,:])

        return image,labels


    def get_transform(self):

        self.image = self.random_brightness(self.image,low = -1,high = 1)

        self.image = self.random_contrast(self.image,low = 0.5,high = 1.5)

        self.image = self.random_gamma(self.image,low = 0.5,high = 1.5)

        #self.image,self.labels = self.random_flip(self.image,self.labels)

        self.image = self.random_blur(self.image,low = 0, high = 2)

        return self.image,self.labels



def make_dataset(path,batch_size):

    os.path.abspath(path)

    data_path = os.path.join(path, 'img_lbl/')

    num_data = len(os.listdir(data_path))

    print('{} images found in {}'.format(num_data, data_path))

    data_path = os.path.join(data_path,str('*.npy'))

    dataset = tf.data.Dataset.list_files(data_path)

    dataset = dataset.shuffle(buffer_size = num_data).batch(batch_size = batch_size)

    num_batches = num_data // batch_size

    return dataset,num_batches


def make_batch(batch,config):

    batch = np.array(batch)

    batch_images = []

    batch_labels = []

    for i in range(len(batch)):

        data = np.load(batch[i])

        img = data[:,:,:,0]
        img = img.reshape(img.shape+(1,)).astype(np.float32)

        lbl = data[:,:,:,1:]
        # remove oxygen, element at index 4 in the labels

        if config.model_kwargs.remove_O:
            lbl = lbl[:,:,:,:2]

        rnd_img = Random_Imaging(image = img,labels = lbl)
        img,lbl = rnd_img.get_transform()

        batch_images.append(img)

        batch_labels.append(lbl)

    batch_images = np.concatenate(batch_images)

    batch_labels = np.concatenate(batch_labels)

    return  [batch_images, batch_labels]


