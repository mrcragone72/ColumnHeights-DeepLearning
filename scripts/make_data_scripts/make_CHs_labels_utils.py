import numpy as np

from scipy.spatial.distance import cdist



class HEA_Labels(object):

    def __init__(self,HEA_model,image_size,resolution,spot_size):

        self.HEA_model = HEA_model

        self.image_size = image_size

        self.resolution = resolution
        
        self.spot_size = spot_size

    def get_labels_single_element(self,element):

        positions = self.HEA_model.get_positions()[np.array(self.HEA_model.get_chemical_symbols()) == element][:,:2]/self.resolution

        width = int(self.spot_size/self.resolution)

        x, y = np.mgrid[0:self.image_size[0], 0:self.image_size[1]]

        labels = np.zeros(self.image_size)

        for p in (positions):

            p_round = np.round(p).astype(int)

            min_xi = np.max((p_round[0] - width * 4, 0))
            max_xi = np.min((p_round[0] + width * 4 + 1, self.image_size[0]))
            min_yi = np.max((p_round[1] - width * 4, 0))
            max_yi = np.min((p_round[1] + width * 4 + 1, self.image_size[1]))

            xi = x[min_xi:max_xi, min_yi:max_yi]
            yi = y[min_xi:max_xi, min_yi:max_yi]

            v = np.array([xi.ravel(), yi.ravel()])

            labels[xi, yi] += np.exp(-cdist([p], v.T) ** 2 / (2 * width ** 2)).reshape(xi.shape)

        return labels

    def get_labels_multi_elements(self):

        labels_all_elements = []

        for element in np.unique(self.HEA_model.get_chemical_symbols()):

            labels_single_element = self.get_labels_single_element(element)

            labels_single_element = labels_single_element.reshape(labels_single_element.shape + (1,))

            labels_all_elements.append(labels_single_element)

        labels_all_elements = np.concatenate(labels_all_elements, axis = 2)

        return labels_all_elements



        





