#!/usr/bin/python

'''
Author: Marco Ragone, Computational Multiphase Transport Laboratory, University of Illinois at Chicago
'''

import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as mixed_precision

"""
The implemented deep learning (DL) model is a fully convolutional network (FCN) characherized by
a symmetrical structure. The encoder and the decoder have 3 blocks of convolutional layers with 3x3 filters.
Between the encoder and the decoder, a bridge of convolutional layer is present. Each convolutional block
incorporates an input Conv2D layer, a residual block of 3 Conv2D layers and an output Conv2D layer.

The number of output channels in the convolutional blocks are 32,64 and 128 rispectively, 
while the bridge has 256 output channels. The last convolutional layer of the network is the score layer, 
with 1x1 filters and 5 output channels.

Each convolutional layer includes Batch Normalization and PReLU activation function.

In the encoder, Max-Pooling layers are used to halve the size of the image, while in the decoder the original
image size is restored using Bilinear Upsampling.

Skip connections are used to connect the convolutional blocks in the encoder and 
the symmetrical convolutional blocks in the decoder.

"""

mp = False

if mp:

    policy = mixed_precision.Policy('mixed_float16')

    mixed_precision.set_policy(policy)

weight_decay=True

def activation_function(x,name=None):

    act_fnct = tf.keras.layers.PReLU(shared_axes=[1,2], name=name+"/PReLU", alpha_initializer=tf.keras.initializers.Constant(0.01))

    return act_fnct(x)

def regularization():
    if weight_decay:
        return tf.keras.regularizers.l2(weight_decay)
    else:
        return None

def convolutional_layer(x, channels, kernel_size=3, name='convolutional_layer'):

    convolutional_layer = tf.keras.layers.Conv2D(channels, kernel_size, padding='same',
                             kernel_regularizer=regularization(),
                             kernel_initializer='RandomNormal',
                             bias_initializer=tf.keras.initializers.Constant(0.1),
                             name=name)

    x = activation_function(convolutional_layer(x),name=name)

    x = tf.keras.layers.BatchNormalization(name=name+'_batch_normalization')(x)

    return x

def skip_connection(x, y,name = 'element_wise_addition'):
    return tf.keras.layers.add([x, y], name = name)

def residual_block(x, channels, name='residual_block'):

    y = convolutional_layer(x, channels, name=name+'/convolution_in_residual_1')
    y = convolutional_layer(y, channels, name=name+'/convolution_in_residual_2')
    y = convolutional_layer(y, channels, name=name+'/convolution_in_residual_3')
    connection_x_y = skip_connection(x, y, name=name+'/element_wise_addition')

    return connection_x_y

def convolution_block(x, channels, name='convolution_block'):

    x = convolutional_layer(x, channels, name=name+"/convolution_1")
    x = residual_block(x, channels, name=name+"/residual_block")
    x = convolutional_layer(x,channels, name=name+"/convolution_2")

    return(x)


def max_pooling_layer(x, name='max_pooling'):

    x =  tf.keras.layers.MaxPooling2D(pool_size=2, padding='same', name=name)(x)

    return x



def upsampling_layer(x, channels, name='upsampling_layer'):

    x = tf.keras.layers.UpSampling2D(name=name+'/upsampling')(x)
    x = convolutional_layer(x, channels, kernel_size=1, name=name+'/upsampling_convolutional')
    return (x)

def score_layer(x, channels, kernel_size=1, name="score_layer"):

    final_convolution = tf.keras.layers.Conv2D(channels, kernel_size,
                             activation='relu',
                             padding='same',
                             kernel_regularizer=regularization(),
                             kernel_initializer='RandomNormal',
                             bias_initializer=tf.keras.initializers.Constant(0.1),
                             name=name)

    final_activation = tf.keras.layers.Activation('relu', dtype='float32')

    return final_activation(final_convolution(x))


def FCN(input_tensor, output_channels, channels = 32):


    # encoder : the convolution takes action

    # block 1
    # in: 256 x 256 x 1
    # out 128 x 128 x 32
    down_sampling_block_1 = convolution_block(input_tensor, channels, name="down_sampling_block_1")
    pooling_1 = max_pooling_layer(down_sampling_block_1, name="max_pooling_1")

    # block 2
    # in: 128 x 128 x 32
    # out: 64 x 64 x 64
    down_sampling_block_2 = convolution_block(pooling_1, channels*2, name="down_sampling_block_2")
    pooling_2 = max_pooling_layer(down_sampling_block_2, name="max_pooling_2")

    # block 3
    # in: 64 x 64 x 64
    # out: 32  x 32 x 128
    down_sampling_block_3 = convolution_block(pooling_2, channels*4, name="down_sampling_block_3")
    pooling_3 = max_pooling_layer(down_sampling_block_3, name="max_pooling_3")

    # block 4
    # in: 32 x 32 x 128
    # out: 16 x 16 x 256
    down_sampling_block_4 = convolution_block(pooling_3, channels * 8, name="down_sampling_block_4")
    pooling_4 = max_pooling_layer(down_sampling_block_4, name="max_pooling_4")

    # block 5
    # in: 16 x 16 x 256
    # out: 8  x 8 x 512
    down_sampling_block_5 = convolution_block(pooling_4, channels * 16, name="down_sampling_block_5")
    pooling_5 = max_pooling_layer(down_sampling_block_5, name="max_pooling_5")

    # bridge layer between the encoder and the decoder
    # in: 8 x 8 x 512
    # out: 8 x 8 x 1024
    bridge_layer = convolution_block(pooling_5, channels*32, name="bridge_layer")

    # decoder : the deconvolution takes action

    # simmetry with block 5
    # in: 8 x 8 x 1024
    # out: 16  x 16 x 512
    up_sampling_block_5 = upsampling_layer(bridge_layer, channels * 16, name="upsampling_5_in")
    up_sampling_block_5_connected = skip_connection(up_sampling_block_5, down_sampling_block_5,name='skip_connection_5')
    up_sampling_block_5_convolution = convolution_block(up_sampling_block_5_connected, channels * 16, name="up_sampling_5_out")

    # simmetry with block 4
    # in: 16 x 16 x 512
    # out: 32  x 32 x 256
    up_sampling_block_4 = upsampling_layer(up_sampling_block_5_convolution, channels * 8, name="upsampling_4_in")
    up_sampling_block_4_connected = skip_connection(up_sampling_block_4, down_sampling_block_4,name='skip_connection_4')
    up_sampling_block_4_convolution = convolution_block(up_sampling_block_4_connected, channels * 8,name="up_sampling_4_out")

    # simmetry with block 3
    # in: 32 x 32 x 256
    # out: 64  x 64 x 128
    up_sampling_block_3 = upsampling_layer(up_sampling_block_4_convolution, channels*4, name="upsampling_3_in")
    up_sampling_block_3_connected = skip_connection(up_sampling_block_3, down_sampling_block_3, name='skip_connection_3')
    up_sampling_block_3_convolution = convolution_block(up_sampling_block_3_connected, channels*4, name="up_sampling_3_out")

    # simmetry with block 2
    # in: 64 x 64 x 128
    # out: 128  x 128 x 64
    up_sampling_block_2 = upsampling_layer(up_sampling_block_3_convolution, channels*2, name="upsampling_2_in")
    up_sampling_block_2_connected = skip_connection(up_sampling_block_2, down_sampling_block_2, name='skip_connection_2')
    up_sampling_block_2_convolution = convolution_block(up_sampling_block_2_connected, channels*2, name="up_sampling_2_out")

    # simmetry with block 1
    # in: 128 x 128 x 64
    # out: 256  x 256 x 32
    up_sampling_block_1 = upsampling_layer(up_sampling_block_2_convolution, channels, name="upsampling_1_in")
    up_sampling_block_1_connected = skip_connection(up_sampling_block_1, down_sampling_block_1, name='skip_connection_1')
    up_sampling_block_1_convolution = convolution_block(up_sampling_block_1_connected, channels, name="up_sampling_1_out")

    # final prediction
    # in: 256 x 256 x 32
    # out: 256  x 256 x 3
    inference = score_layer(up_sampling_block_1_convolution, channels=output_channels)

    return tf.keras.Model(input_tensor,inference)
