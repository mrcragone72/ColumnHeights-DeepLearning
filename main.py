#!/usr/bin/python

'''
Author: Marco Ragone, Computational Multiphase Transport Laboratory, University of Illinois at Chicago
'''

import subprocess
from argparse import ArgumentParser

packages = ['scikit-learn','scikit-image', 'pandas']

for p in packages:
	print('Installing {}'.format(p))
	print('')
	subprocess.call(['pip', 'install', '{}'.format(p)])
	print('')

print('Libraries installation Done!')
print('')

import sys
sys.path.append('./scripts/')
sys.path.append('./scripts/make_data_scripts/')
sys.path.append('./scripts/training_scripts/')


from config import Config
from training_utils import*
from make_HEA_data_utils import*

if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument('-j', type=str, dest='json_path', help='Path to configuration .json file')
	args = parser.parse_args()

	config = Config(args.json_path)

	# for creating data

	#make_HEA_data(config)
	#make_HEA_data_multiprocessing(config)

	# for training

	train_serial(config)
	#train_model_parallel(config)
	#train_data_parallel(config)
