import numpy as np

import dm4reader
import cv2

from skimage.filters import gaussian
from skimage.feature import peak_local_max

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def read_dm4(file_path):
    dm4data = dm4reader.DM4File.open(file_path)

    tags = dm4data.read_directory()

    image_data_tag = tags.named_subdirs['ImageList'].unnamed_subdirs[1].named_subdirs['ImageData']
    image_tag = image_data_tag.named_tags['Data']

    XDim = dm4data.read_tag_data(image_data_tag.named_subdirs['Dimensions'].unnamed_tags[0])
    YDim = dm4data.read_tag_data(image_data_tag.named_subdirs['Dimensions'].unnamed_tags[1])

    img = np.array(dm4data.read_tag_data(image_tag), dtype=np.uint16)
    img = np.reshape(img, (YDim, XDim))

    return img


def read_exp_img(file_path, file_format='tif'):
    if file_format == 'dm4':

        exp_img = read_dm4(file_path=file_path)

        print('Raw experimental image shape: {}'.format(exp_img.shape))

    else:

        exp_img = cv2.imread(file_path)

        print('Raw experimental image shape: {}'.format(exp_img.shape))

        if len(exp_img.shape) == 3:
            print('Converting BGR image to gray scale image')
            exp_img = cv2.cvtColor(exp_img, cv2.COLOR_BGR2GRAY)

            print('Gray scale image shape: {}'.format(exp_img.shape))

            #exp_img = cv2.resize(exp_img, (512, 512))

            #print('Gray scale image shape: {}'.format(exp_img.shape))

    return exp_img


def get_local_normalization(img, resolution):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))

    img = img - gaussian(img, 12 * int(1 / resolution))

    img = img / np.sqrt(gaussian(img ** 2, 12 * int(1 / resolution)))

    return img


def plot_exp_img_predictions(exp_img,
                             pred_exp_img,
                             chemical_elements,
                             n1_x,
                             delta_x,
                             n1_y,
                             delta_y,
                             zoom=False,
                             conc=False,
                             interpolation=None):
    
    pred_exp_img = pred_exp_img[0, :, :, :]

    if zoom:
        exp_img = exp_img[n1_y:n1_y + delta_y,
                  n1_x:n1_x + delta_x]

        pred_exp_img = pred_exp_img[n1_y:n1_y + delta_y,
                       n1_x:n1_x + delta_x, :]

    fig = plt.figure(figsize=(14, 14))

    ax = fig.add_subplot(2, 3, 1)
    im = ax.imshow(exp_img, cmap='gray', interpolation=interpolation)
    plt.title('Experimental STEM image', fontsize=20)
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax1)

    CH_max = np.max(pred_exp_img)
    thrsld = -10

    pred_all = np.zeros((pred_exp_img.shape[0], pred_exp_img.shape[1]))
    
    fig = plt.figure(figsize=(14, 14))

    for i in range(len(chemical_elements)):

        pred = np.array(pred_exp_img[:, :, i])
        pred[pred < thrsld] = 0
        
        if chemical_elements[i] is not 'O':
            pred_all += pred

        ax = fig.add_subplot(3, 3, i + 1)
        if conc:
            im = ax.imshow(pred / CH_max, vmin=0, vmax=1, cmap='jet', interpolation=interpolation)
        else:
            #im = ax.imshow(pred, vmin=0, vmax=CH_max, cmap='jet', interpolation=interpolation)
            im = ax.imshow(pred, vmin=0, cmap='jet', interpolation=interpolation)
        plt.title('{} CHs Prediction'.format(chemical_elements[i]), fontsize=20)
        
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax1)

    fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot(1, 1, 1)
    if conc:
        im = ax.imshow(pred_all / np.max(pred_all), vmin=0, cmap='jet', interpolation=interpolation)
    else:
        im = ax.imshow(pred_all, vmin=0, cmap='jet', interpolation=interpolation)

    plt.title('Total CHs', fontsize=20)
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax1)

