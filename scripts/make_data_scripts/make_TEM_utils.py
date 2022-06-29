import numpy as np

from skimage.filters import gaussian
from skimage.feature import peak_local_max

from pyqstem.imaging import CTF



class SimSTEM(object):

    def __init__(self,
                 qstem,
                 atomic_model,
                 image_size,
                 resolution,
                 probe,
                 slice_thickness,
                 v0,
                 alpha,
                 defocus,
                 Cs,
                 astig_mag,
                 astig_angle,
                 random_aberrations,
                 add_local_norm,
                 add_noise,
                 noise_mean,
                 noise_std,
                 noise_window_size):

        self.qstem = qstem

        self.atomic_model = atomic_model

        self.image_size = image_size

        self.resolution = resolution

        self.probe = probe

        self.slice_thickness = slice_thickness

        self.v0 = v0

        self.alpha = alpha

        self.defocus = defocus

        self.Cs = Cs

        self.astig_mag = astig_mag

        self.astig_angle = astig_angle
        
        self.random_aberrations = random_aberrations
        
        self.add_local_norm = add_local_norm

        self.add_noise = add_noise

        self.noise_mean = noise_mean

        self.noise_std = noise_std
        
        self.noise_window_size = noise_window_size
        


    def get_DF_img(self):

        self.qstem.set_atoms(self.atomic_model)

        self.qstem.build_probe(
                               v0 = self.v0,
                               alpha = self.alpha,
                               num_samples = (self.probe, self.probe),
                               resolution = (self.resolution,self.resolution),
                               Cs = self.Cs,
                               defocus = self.defocus,
                               astig_mag = self.astig_mag,
                               astig_angle = self.astig_angle,
                               aberrations = self.random_aberrations
                                )

        self.wave = self.qstem.get_wave()

        self.num_slices = self.probe / self.slice_thickness

        self.spatial_domain =  (self.image_size[0] * self.resolution, self.image_size[1] * self.resolution)

        self.scan_range = [[0, self.spatial_domain[0], self.image_size[0]],
                          [0, self.spatial_domain[1], self.image_size[1]]]

        self.qstem.build_potential(num_slices = self.num_slices, scan_range = self.scan_range)

        detector1_radii = (70,200)

        detector2_radii = (0, 70)

        self.qstem.add_detector('detector1', detector1_radii)

        self.qstem.add_detector('detector2', detector2_radii)

        self.qstem.run()

        self.img = self.qstem.read_detector('detector2') * (-1) 

        return self.img


    def get_local_normalization(self):

        self.img = (self.img - np.min(self.img)) / (np.max(self.img) - np.min(self.img))

        self.img = self.img - gaussian(self.img, 12 / self.resolution)

        self.img = self.img / np.sqrt(gaussian(self.img ** 2, 12 / self.resolution))

        return self.img


    def get_local_noise(self):

        self.threshold_abs = 1e-6

        self.peaks = peak_local_max(self.img, min_distance = int(1/self.resolution), threshold_abs = self.threshold_abs)


        for _, p in enumerate(self.peaks):

            for i in range(p[0] - self.noise_window_size // 2, p[0] + self.noise_window_size// 2):

                for j in range(p[1] - self.noise_window_size // 2, p[1] + self.noise_window_size // 2):

                    if (i > 0 and j > 0 and i > 0 and j < self.image_size[1] and i < self.image_size[0] and j > 0 and i < self.image_size[0] and j < self.image_size[1]):

                        self.img[i][j] = self.img[i][j] + np.random.normal(self.noise_mean, self.noise_std, (1, 1))

        self.img[self.img <-1] = -1

        return self.img

    def get_img(self):

        self.get_DF_img()
        
        if self.add_local_norm:

            self.get_local_normalization()

        if self.add_noise:

            self.get_local_noise()

        return self.img


class SimHRTEM(object):

    def __init__(self,
                 qstem,
                 atomic_model,
                 image_size,
                 resolution,
                 Cs,
                 defocus,
                 focal_spread,
                 aberrations,
                 blur,
                 dose,
                 MTF_param,
                 add_local_norm,
                 add_noise,
                 noise_mean,
                 noise_std,
                 noise_window_size
                 ):

        self.qstem = qstem

        self.atomic_model = atomic_model

        self.image_size = image_size

        self.resolution = resolution

        self.Cs = Cs

        self.defocus = defocus

        self.focal_spread = focal_spread

        self.aberrations = aberrations

        self.blur = blur

        self.dose = dose

        self.MTF_param = MTF_param

        self.add_local_norm = add_local_norm

        self.add_noise = add_noise

        self.noise_mean = noise_mean

        self.noise_std = noise_std

        self.noise_window_size = noise_window_size

    def get_HRTEM_img(self):

        self.qstem.set_atoms(self.atomic_model)

        self.wave_size = (int(self.NP_model.get_cell()[0, 0] / self.resolution),
                          int(self.NP_model.get_cell()[1, 1] / self.resolution))

        self.qstem.build_wave('plane', 300, self.wave_size)
        self.qstem.build_potential(int(self.NP_model.get_cell()[2, 2] * 2))
        self.qstem.run()

        wave = self.qstem.get_wave()

        wave.array = wave.array.astype(np.complex64)

        self.ctf = CTF(defocus=self.defocus,
                       Cs=self.Cs,
                       focal_spread=self.focal_spread,
                       aberrations=self.aberrations)

        self.img = wave.apply_ctf(self.ctf).detect(resample=self.resolution,
                                                   blur=self.blur,
                                                   dose=self.dose,
                                                   MTF_param=self.MTF_param)

        return self.img

    def get_local_normalization(self):

        self.img = (self.img - np.min(self.img)) / (np.max(self.img) - np.min(self.img))

        self.img = self.img - gaussian(self.img, 12 / self.resolution)

        self.img = self.img / np.sqrt(gaussian(self.img ** 2, 12 / self.resolution))

        return self.img

    def get_local_noise(self):

        self.threshold_abs = 1e-6

        self.peaks = peak_local_max(self.img, min_distance=int(1 / self.resolution), threshold_abs=self.threshold_abs)

        for _, p in enumerate(self.peaks):

            for i in range(p[0] - self.noise_window_size // 2, p[0] + self.noise_window_size // 2):

                for j in range(p[1] - self.noise_window_size // 2, p[1] + self.noise_window_size // 2):

                    if (i > 0 and j > 0 and i > 0 and j < self.image_size[1] and i < self.image_size[
                        0] and j > 0 and i < self.image_size[0] and j < self.image_size[1]):
                        self.img[i][j] = self.img[i][j] + np.random.normal(self.noise_mean, self.noise_std, (1, 1))

        self.img[self.img < -1] = -1

        return self.img

    def get_img(self):

        self.get_HRTEM_img()

        if self.add_local_norm:

            self.get_local_normalization()

        if self.add_noise:

            self.get_local_noise()

        return self.img
