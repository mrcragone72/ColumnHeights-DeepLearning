import numpy as np
import os
import matplotlib.pyplot as plt

from skimage.feature import peak_local_max
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def make_folder(path):
    if path and not os.path.exists(path):
        os.makedirs(path)
        print('Creating directory at {}'.format(path))


def get_peaks_pos(image, min_distance, threshold_abs):
    peaks_pos = peak_local_max(image, min_distance=min_distance, threshold_abs=threshold_abs)

    return peaks_pos


def get_CHs_I_corr_0(img, lbl, min_distance, threshold_abs, include_CHs_0=True):
    peaks_pos_all = get_peaks_pos(np.sum(lbl, axis=2), min_distance=min_distance, threshold_abs=threshold_abs)

    I_all_elements = []
    CHs_all_elements = []
    slope_all_elements = []
    intercept_all_elements = []
    R2_all_elements = []
    slope_all_elements = []

    # STEM intensity - column heights correlation by element
    for ce in range(lbl.shape[2]):

        if include_CHs_0:

            peaks_pos = peaks_pos_all

        else:

            peaks_pos = get_peaks_pos(lbl[:, :, ce], min_distance=min_distance, threshold_abs=threshold_abs)

        I = img[peaks_pos[:, 0], peaks_pos[:, 1]]
        CHs = np.round(lbl[peaks_pos[:, 0], peaks_pos[:, 1], ce])

        lr = LinearRegression()
        lr.fit(CHs.reshape(-1, 1), I.reshape(-1, 1))
        slope = lr.coef_[0][0]
        intercept = lr.intercept_[0]

        # R2 = r2_score(I,lr.predict(CHs.reshape(-1,1)))
        R2 = r2_score(I, slope * CHs + intercept)

        # R2 = lr.score(I.reshape(-1,1),
        #               CHs.reshape(-1,1))

        I_all_elements.append(I)
        CHs_all_elements.append(CHs)

        slope_all_elements.append(slope)
        intercept_all_elements.append(intercept)

        R2_all_elements.append(R2)

    # STEM intensity - total column heights correlation
    peaks_pos = get_peaks_pos(np.sum(lbl, axis=2), min_distance=min_distance, threshold_abs=threshold_abs)
    I = img[peaks_pos[:, 0], peaks_pos[:, 1]]
    CHs = np.round(np.sum(lbl, axis=2)[peaks_pos[:, 0], peaks_pos[:, 1]])

    lr = LinearRegression()
    lr.fit(CHs.reshape(-1, 1), I.reshape(-1, 1))
    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]

    # R2 = lr.score(I.reshape(-1,1),
    #               CHs.reshape(-1,1))

    I_all_elements.append(I)
    CHs_all_elements.append(CHs)

    # slope_all_elements.append(slope)
    intercept_all_elements.append(intercept)

    # R2_all_elements.append(R2)

    return CHs_all_elements, I_all_elements, slope_all_elements, intercept_all_elements, R2_all_elements


def plot_CHs_I_corr(chemical_symbols, CHs_all_elements, I_all_elements, slope_all_elements, intercept_all_elements,
                    R2_all_elements):
    colors = ['brown', 'orange', 'purple', 'green', 'red', 'blue']

    fig = plt.figure(figsize=(20, 20))

    for cs in range(len(chemical_symbols)):

        ax = fig.add_subplot(3, 2, cs + 1)
        ax.scatter(CHs_all_elements[cs],
                   I_all_elements[cs],
                   c=colors[cs])
        ax.plot(CHs_all_elements[cs],
                slope_all_elements[cs] * CHs_all_elements[cs] + intercept_all_elements[cs],
                c=colors[cs])
        ax.set_ylim(-2, 11)

        if cs < len(chemical_symbols):
            plt.title(chemical_symbols[cs])
        else:
            plt.title('Total')

        plt.xlabel('Column Height')
        plt.ylabel('Column Intensity')

        # plt.legend(['slope = {:.2f}, R2 = {:.2f}'.format(slope_all_elements[cs],
        #                                                R2_all_elements[cs])])

        plt.legend(['R2 = {:.2f}'.format(R2_all_elements[cs])])


def get_CHs_I_corr(img,lbl, include_CHs_0=True):

    peaks_pos_all = peak_local_max(np.sum(lbl, axis=2))

    I_all_elements = []
    CHs_all_elements = []
    slope_all_elements = []
    intercept_all_elements = []
    R2_all_elements = []

    # STEM intensity - column heights correlation by element
    for ce in range(lbl.shape[2]):

        if include_CHs_0:

            peaks_pos = peaks_pos_all

        else:

            peaks_pos = peak_local_max(lbl[:, :, ce])

        I = img[peaks_pos[:, 0], peaks_pos[:, 1]]
        CHs = np.round(lbl[peaks_pos[:, 0], peaks_pos[:, 1], ce])

        lr = LinearRegression()
        lr.fit(CHs.reshape(-1, 1), I.reshape(-1, 1))
        slope = lr.coef_[0][0]
        intercept = lr.intercept_[0]

        # R2 = r2_score(I,lr.predict(CHs.reshape(-1,1)))
        R2 = r2_score(I, slope * CHs + intercept)

        # R2 = lr.score(I.reshape(-1,1),
        #               CHs.reshape(-1,1))

        I_all_elements.append(I)
        CHs_all_elements.append(CHs)

        slope_all_elements.append(slope)
        intercept_all_elements.append(intercept)

        R2_all_elements.append(R2)

    # STEM intensity - total column heights correlation
    peaks_pos = peak_local_max(np.sum(lbl, axis=2))
    I = img[peaks_pos[:, 0], peaks_pos[:, 1]]
    CHs = np.round(np.sum(lbl, axis=2)[peaks_pos[:, 0], peaks_pos[:, 1]])

    lr = LinearRegression()
    lr.fit(CHs.reshape(-1, 1), I.reshape(-1, 1))
    slope = lr.coef_[0][0]
    intercept = lr.intercept_[0]

    # R2 = lr.score(I.reshape(-1,1),
    #               CHs.reshape(-1,1))

    I_all_elements.append(I)
    CHs_all_elements.append(CHs)

    #slope_all_elements.append(slope)
    intercept_all_elements.append(intercept)

    # R2_all_elements.append(R2)

    return CHs_all_elements, I_all_elements, slope_all_elements, intercept_all_elements, R2_all_elements


data_folder_path = './MnFeNiCuZnO_data_-110_noise/'


train = True

if train:
    img_lbl_folder_path = os.path.join(data_folder_path, 'training_data/img_lbl/')
    plot_path = os.path.join(data_folder_path, 'R2-slope-I-CHs-corr_plot/train/')
else:
    img_lbl_folder_path = os.path.join(data_folder_path, 'test_data/img_lbl/')
    plot_path = os.path.join(data_folder_path, 'R2-slope-I-CHs-corr_plot/test/')

make_folder(plot_path)

chemical_symbols = ['Cu', 'Fe', 'Mn', 'Ni', 'Zn']


R2_all_elements_all_data = [[], [], [], [], []]
slope_all_elements_all_data = [[],[],[],[],[]]
for img_lbl_path in os.listdir(img_lbl_folder_path):

    HEO_img_lbl = np.load(os.path.join(img_lbl_folder_path, img_lbl_path))

    img = HEO_img_lbl[0, :, :, 0]

    lbl = HEO_img_lbl[:, :, :, 1:]
    # remove oxygen, element at index 4 in the labels
    lbl1 = lbl[0, :, :, :4]
    lbl2 = lbl[0, :, :, 5]
    lbl2 = np.expand_dims(lbl2, axis=2)
    lbl = np.concatenate([lbl1, lbl2], axis=2)

    _, _, slope_all_elements, _, R2_all_elements = get_CHs_I_corr(img,
                                                                 lbl,
                                                                 include_CHs_0 = True)

    for i, r2 in enumerate(R2_all_elements):
        R2_all_elements_all_data[i].append(r2)

    for i, slope in enumerate(slope_all_elements):
        slope_all_elements_all_data[i].append(slope)

fig = plt.figure(figsize=(14, 14))

for i, r2 in enumerate(R2_all_elements_all_data):
    mean = np.mean(r2)
    Q1 = np.percentile(r2, q=25)
    Q3 = np.percentile(r2, q=75)

    ax = fig.add_subplot(3, 2, i + 1)
    plt.boxplot(r2)
    plt.title(chemical_symbols[i])
    plt.legend(['mean R2 = {:.4f}'.format(mean),
                'Q1 - 1.5 x Q1 = {:.4f}'.format(Q1 - 1.5 * Q1),
                'Q1 = {:.4f}'.format(Q1),
                'Q3 = {:.4f}'.format(Q3),
                'Q3 + 1.5 x Q3 = {:.4f}'.format(Q3 + 1.5 * Q3)])

fig.savefig(os.path.join(plot_path,'R2-plot-1.png'), bbox_inches='tight')
print('R2 plot 1 saved at {}'.format(os.path.join(plot_path,'R2-plot-1.png')))


fig = plt.figure(figsize=(14, 7))
plt.boxplot(R2_all_elements_all_data)
fig.savefig(os.path.join(plot_path,'R2-plot-2.png'), bbox_inches='tight')
print('R2 plot 2 saved at {}'.format(os.path.join(plot_path,'R2-plot-2.png')))

fig = plt.figure(figsize=(14, 14))

for i, slope in enumerate(slope_all_elements_all_data):
    mean = np.mean(slope)
    #Q1 = np.percentile(slope, q=25)
    #Q3 = np.percentile(slope, q=75)

    ax = fig.add_subplot(3, 2, i + 1)
    plt.boxplot(slope)
    plt.title(chemical_symbols[i])
    plt.legend(['mean slope = {:.4f}'.format(mean)])
                #'Q1 - 1.5 x Q1 = {:.4f}'.format(Q1 - 1.5 * Q1),
                #'Q1 = {:.4f}'.format(Q1),
                #'Q3 = {:.4f}'.format(Q3),
                #'Q3 + 1.5 x Q3 = {:.4f}'.format(Q3 + 1.5 * Q3)])

fig.savefig(os.path.join(plot_path,'slope-plot-1.png'), bbox_inches='tight')
print('slope plot 1 saved at {}'.format(os.path.join(plot_path,'slope-plot-1.png')))

fig = plt.figure(figsize=(14, 7))
plt.boxplot(slope_all_elements_all_data)
fig.savefig(os.path.join(plot_path,'slope-plot-2.png'), bbox_inches='tight')
print('slope plot 2 saved at {}'.format(os.path.join(plot_path,'slope-plot-2.png')))

print('Done!')
