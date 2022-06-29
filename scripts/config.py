#!/usr/bin/python

'''
Author: Marco Ragone, Computational Multiphase Transport Laboratory, University of Illinois at Chicago
'''


import os
import json
import pandas as pd

from datetime import datetime

import tensorflow as tf
#import horovod.tensorflow as hvd

def make_folder(path):
    if path and not os.path.exists(path):
        os.makedirs(path)
        print('Creating directory at {}'.format(path))

    else:

        print('Directory {} already exists!'.format(path))


def Config(json_path):

    with open(json_path) as json_path:

        json_dict = json.load(json_path)

    config = pd.DataFrame(json_dict)

    path = os.path.join('experiments/',
                        config.data_kwargs.experiment + '/')

    path = os.path.join(path, 'json_templates/')

    make_folder(path)


    now = datetime.now()
    now = now.strftime("%m-%d-%Y_%H-%M-%S")

    print('Saving config_{}.json in {}'.format(now,path))

    with open(os.path.join(path,'config_{}.json').format(now), 'w') as json_path:
        json.dump(json_dict, json_path, indent = 4)

    return config


def make_config_proto():

    config_proto = tf.compat.v1.ConfigProto()
    config_proto.allow_soft_placement = True
    config_proto.log_device_placement = False
    config_proto.gpu_options.allow_growth = True
    config_proto.gpu_options.force_gpu_compatible = True
    config_proto.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

def make_config_proto_hvd(hvd):

    hvd.init()

    config_proto = tf.compat.v1.ConfigProto()
    config_proto.graph_options.optimizer_options.global_jit_level = tf.compat.v1.OptimizerOptions.ON_1

    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

