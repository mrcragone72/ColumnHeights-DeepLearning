import numpy as np

import os

import platform
import multiprocessing as mp
import itertools


import sys
sys.path.append('./scripts')
from make_TEM_utils import *
from make_structure_utils import *
from make_CHs_labels_utils import *

from pyqstem import PyQSTEM
from ase.io import write


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



def make_folder(path):
    if path and not os.path.exists(path):
        os.makedirs(path)
        print('Creating directory at {}'.format(path))

    else:

        print('Directory {} already exists!'.format(path))



class HEA_Data(object):

    def __init__(self, model, img, lbl, path, data_index):

        self.model = model

        self.img = img

        self.lbl = lbl

        self.path = path

        self.data_index = data_index

    def save_HEA_model(self):

        if self.data_index < 50:
            write(self.path + 'models/HEA_model_{}.xyz'.format(self.data_index), self.model)

    def save_HEA_data(self):

        self.img = self.img.reshape((1,) + self.img.shape + (1,))

        self.lbl = self.lbl.reshape((1,) + self.lbl.shape)

        self.data = np.concatenate([self.img, self.lbl], axis=3)


        np.save(self.path + 'img_lbl/HEA_img_lbl_{}.npy'.format(self.data_index), self.data)

        return self.data

    def save_HEA_plot(self):

        chemical_symbols = list(np.unique(self.model.get_chemical_symbols()))

        fig = plt.figure(figsize=(28, 14))

        ax = fig.add_subplot(2, 4, 1)
        im = ax.imshow(self.img[0, :, :, 0], cmap='gray')
        plt.title('STEM image', fontsize=20)
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax1)

        for cs in range(1, len(chemical_symbols) + 1):
            ax = fig.add_subplot(2, 4, cs + 1)
            im = ax.imshow(self.lbl[0, :, :, cs - 1], cmap='jet')
            plt.title('{} CHs'.format(chemical_symbols[cs - 1], fontsize=20))
            divider = make_axes_locatable(ax)
            cax1 = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax1)

        plt.tight_layout()

        if self.data_index < 50:
            fig.savefig(self.path + 'plots/HEA_plot_{}.png'.format(self.data_index), bbox_inches='tight')

        plt.close(fig)

    def save_HEA(self):

        self.save_HEA_model()

        self.save_HEA_data()

        self.save_HEA_plot()


def make_HEA_data_multiprocessing(config):

    print("Running on host '{}'".format(platform.node()))

    n_processors_all = mp.cpu_count()
    n_processors = config.data_kwargs.n_processors
    # n_processors = n_processors_all

    num_data = config.data_kwargs.num_data

    print("Number of available processors: ", n_processors_all)
    print("Number of used processors: ", n_processors)

    if config.data_kwargs.training_data:

        data_folder_path = os.path.join(config.data_kwargs.data_folder_path,'training_data/')

    else:

        data_folder_path = os.path.join(config.data_kwargs.data_folder_path, 'test_data/')

    make_folder(data_folder_path)
    make_folder(os.path.join(data_folder_path, 'models/'))
    make_folder(os.path.join(data_folder_path,'img_lbl/'))
    make_folder(os.path.join(data_folder_path, 'plots/'))


    print('Saving simulated HEA in the directory {}'.format(data_folder_path))

    print('Creating random HEA models from VTK xyz file..')

    ATK_path = config.data_kwargs.ATK_path

    first_data_index = config.data_kwargs.first_data_index
    num_data = config.data_kwargs.num_data

    spatial_domain = (config.structure_kwargs.spatial_domain, config.structure_kwargs.spatial_domain)

    image_size = (config.STEM_image_kwargs.image_size, config.STEM_image_kwargs.image_size)
    resolution = spatial_domain[0] / image_size[0]
    random_transl_xy = config.structure_kwargs.random_transl_xy
    t_xy = config.structure_kwargs.t_xy
    random_rot_y = config.structure_kwargs.random_rot_y
    alpha_y = config.structure_kwargs.alpha_y
    elements_random_mix = config.structure_kwargs.elements_random_mix

    qstem = PyQSTEM(config.STEM_image_kwargs.QSTEM_mode)
    add_local_norm = config.STEM_image_kwargs.add_local_norm
    add_noise = config.STEM_image_kwargs.add_noise
    noise_mean = config.STEM_image_kwargs.noise_mean
    noise_std = config.STEM_image_kwargs.noise_std
    spot_size = config.STEM_image_kwargs.spot_size  # [nm], columns diameter in the segmentation maps
    noise_window_size = 1 * int(10 / spot_size)

    probe = config.STEM_image_kwargs.probe # probe size [nm]
    slice_thickness = config.STEM_image_kwargs.slice_thickness # [nm]


    global HEA_multiprocessing
    def HEA_multiprocessing(data_index):

        print('Processing HEA [{}/{}]'.format(data_index, config.data_kwargs.num_data))

        structure_path = os.path.join(config.data_kwargs.ATK_path,
                                      'structure_{}.xyz'.format(random.choice(np.arange(1,config.data_kwargs.n_structures + 1))))

        random_HEA = ATK_Random_HEA(structure_path,
                                    spatial_domain,
                                    random_transl_xy,
                                    t_xy,
                                    random_rot_y,
                                    alpha_y)

        random_HEA.get_model()

        random_HEA.random_mix(elements_random_mix)

        random_HEA_model = random_HEA.model

        random_v0 = np.random.uniform(config.STEM_image_kwargs.v0[0],
                                      config.STEM_image_kwargs.v0[1])  # acceleration voltage [keV]

        random_alpha = np.random.uniform(config.STEM_image_kwargs.alpha[0],
                                         config.STEM_image_kwargs.alpha[1])  # convergence_angle [mrad]

        random_defocus = np.random.uniform(config.STEM_image_kwargs.defocus[0],
                                           config.STEM_image_kwargs.defocus[1])  # defocus [A]

        random_Cs = np.random.uniform(config.STEM_image_kwargs.Cs[0],
                                      config.STEM_image_kwargs.Cs[1])  # 1st order aberration

        random_astig_mag = np.random.uniform(config.STEM_image_kwargs.astig_mag[0],
                                             config.STEM_image_kwargs.astig_mag[1])  # astigmation magnitude [A]

        random_astig_angle = np.random.uniform(config.STEM_image_kwargs.astig_angle[0],
                                               config.STEM_image_kwargs.astig_angle[1])  # astigmation angle [A]

        random_aberrations = {'a33': config.STEM_image_kwargs.a33,
                              'phi33': config.STEM_image_kwargs.phi33}

        HEA_stem = SimSTEM(qstem,
                            random_HEA_model,
                            image_size,
                            resolution,
                            probe,
                            slice_thickness,
                            random_v0,
                            random_alpha,
                            random_defocus,
                            random_Cs,
                            random_astig_mag,
                            random_astig_angle,
                            random_aberrations,
                            add_local_norm,
                            add_noise,
                            noise_mean,
                            noise_std,
                            noise_window_size)

        img = HEA_stem.get_img()

        HEA_labels = HEA_Labels(random_HEA_model,
                                image_size,
                                resolution,
                                spot_size)

        lbl = HEA_labels.get_labels_multi_elements()

        HEA_data = HEA_Data(random_HEA_model,
                            img,
                            lbl,
                            data_folder_path,
                            data_index)

        HEA_data.save_HEA()


    for counter in range(first_data_index,num_data,n_processors):

        pool = mp.Pool(n_processors)
        pool.map(HEA_multiprocessing,[data_index for data_index in range(counter,counter + n_processors)])
        pool.close()

def make_HEA_data(config):

    if config.data_kwargs.training_data:

        data_folder_path = os.path.join(config.data_kwargs.data_folder_path, 'training_data/')

    else:

        data_folder_path = os.path.join(config.data_kwargs.data_folder_path, 'test_data/')

    make_folder(data_folder_path)
    make_folder(os.path.join(data_folder_path, 'models/'))
    make_folder(os.path.join(data_folder_path, 'img_lbl/'))
    make_folder(os.path.join(data_folder_path, 'plots/'))

    ATK_path = config.data_kwargs.ATK_path

    first_data_index = config.data_kwargs.first_data_index
    num_data = config.data_kwargs.num_data

    spatial_domain = (config.structure_kwargs.spatial_domain,config.structure_kwargs.spatial_domain)

    image_size = (config.STEM_image_kwargs.image_size, config.STEM_image_kwargs.image_size)
    resolution = spatial_domain[0] / image_size[0]
    random_transl_xy = config.structure_kwargs.random_transl_xy
    t_xy = config.structure_kwargs.t_xy
    random_rot_y = config.structure_kwargs.random_rot_y
    alpha_y = config.structure_kwargs.alpha_y
    elements_random_mix = config.structure_kwargs.elements_random_mix

    qstem = PyQSTEM(config.STEM_image_kwargs.QSTEM_mode)
    add_local_norm = config.STEM_image_kwargs.add_local_norm
    add_noise = config.STEM_image_kwargs.add_noise
    noise_mean = config.STEM_image_kwargs.noise_mean
    noise_std = config.STEM_image_kwargs.noise_std
    spot_size = config.STEM_image_kwargs.spot_size  # [nm], columns diameter in the segmentation maps
    noise_window_size = 1 * int(10 / spot_size)

    probe = config.STEM_image_kwargs.probe  # probe size [nm]
    slice_thickness = config.STEM_image_kwargs.slice_thickness  # [nm]


    print('Saving simulated HEO in the directory {}'.format(data_folder_path))

    print('Creating random HEA models from VTK xyz file..')

    for data_index in range(first_data_index, num_data):
        print('Processing HEA [{}/{}]'.format(data_index, num_data))

        structure_path = os.path.join(config.data_kwargs.ATK_path,
                                      'structure_{}.xyz'.format(random.choice(np.arange(1, config.data_kwargs.n_structures + 1))))

        random_HEA = ATK_Random_HEA(structure_path,
                                    spatial_domain,
                                    random_transl_xy,
                                    t_xy,
                                    random_rot_y,
                                    alpha_y)

        random_HEA.get_model()

        random_HEA.random_mix(elements_random_mix)

        random_HEA_model = random_HEA.model

        random_v0 = np.random.uniform(config.STEM_image_kwargs.v0[0],
                                      config.STEM_image_kwargs.v0[1])  # acceleration voltage [keV]

        random_alpha = np.random.uniform(config.STEM_image_kwargs.alpha[0],
                                         config.STEM_image_kwargs.alpha[1])  # convergence_angle [mrad]

        random_defocus = np.random.uniform(config.STEM_image_kwargs.defocus[0],
                                           config.STEM_image_kwargs.defocus[1])  # defocus [A]

        random_Cs = np.random.uniform(config.STEM_image_kwargs.Cs[0],
                                      config.STEM_image_kwargs.Cs[1])  # 1st order aberration

        random_astig_mag = np.random.uniform(config.STEM_image_kwargs.astig_mag[0],
                                             config.STEM_image_kwargs.astig_mag[1])  # astigmation magnitude [A]

        random_astig_angle = np.random.uniform(config.STEM_image_kwargs.astig_angle[0],
                                               config.STEM_image_kwargs.astig_angle[1])  # astigmation angle [A]

        random_aberrations = {'a33': config.STEM_image_kwargs.a33,
                              'phi33': config.STEM_image_kwargs.phi33}


        HEA_stem = SimSTEM(qstem,
                            random_HEA_model,
                            image_size,
                            resolution,
                            probe,
                            slice_thickness,
                            random_v0,
                            random_alpha,
                            random_defocus,
                            random_Cs,
                            random_astig_mag,
                            random_astig_angle,
                            random_aberrations,
                            add_local_norm,
                            add_noise,
                            noise_mean,
                            noise_std,
                            noise_window_size)

        img = HEA_stem.get_img()

        HEA_labels = HEA_Labels(random_HEA_model,
                                image_size,
                                resolution,
                                spot_size)

        lbl = HEA_labels.get_labels_multi_elements()

        HEA_data = HEA_Data(random_HEA_model,
                            img,
                            lbl,
                            data_folder_path,
                            data_index)

        HEA_data.save_HEA()




