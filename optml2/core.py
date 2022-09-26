import numpy as np
import multiprocessing as mp
import tensorflow as tf
from tensorflow import keras
from mdtrack_mobicom19.core import Mdtrack, Parameter
from .data_generator import Generator_1d_tof, Generator_2d_aoa_tof, from_path_to_channel, \
    from_path_to_2d_aoa_tof_sparse_target, \
    from_path_to_2d_aoa_tof_dense_target, map_multiprocess, gen_2d_aoa_tof_physical_paths, gen_1d_tof_physical_paths, \
    from_path_to_1d_tof_dense_target, data_normalization, data_vectorization
from ieee80211.preamble import Preamble
from utils.siggen import Signal
from scipy import ndimage, sparse
import logging
import os
import pickle
import json
import matplotlib.pyplot as pl
from scipy.ndimage.filters import maximum_filter

logger = logging.getLogger(__name__)


class Optml2(object):
    def __init__(self, *args, **kwargs):
        self.debug = kwargs.get('debug', False)
        self.ltf_f = kwargs.get('ltf_f', Preamble.LEGACY_LTF_F_64)
        self.ltf_t = np.fft.ifft(self.ltf_f)
        self.stf_f = kwargs.get('stf_f', Preamble.LEGACY_STF_F_64)
        self.stf_t = kwargs.get('stf_t', Preamble.LEGACY_STF_T_16)
        self.bw = kwargs.get('bw', 20e6)
        self.fc = kwargs.get('fc', 2.412e9)
        self.sp_light = 3e8
        self.tx_antennas = kwargs.get('tx_antennas', 1)
        self.rx_antennas = kwargs.get('rx_antennas', 4)
        self.ant_spacing = kwargs.get('ant_spacing', self.sp_light / self.fc / 2)
        self.fftsize = len(self.ltf_f)
        self.wavelength = self.sp_light / (np.fft.fftfreq(self.fftsize) * self.bw + self.fc)

        # TODO
        self.aoa_cos_search_step = kwargs.get('aoa_cos_search_step', 2 / 200)  # cosine values
        self.aod_search_step = kwargs.get('aod_search_step', 0.02)  # radian
        self.tof_search_step = kwargs.get('tof_search_step', 0.5e-9)  # second
        self.dop_search_step = kwargs.get('dop_search_step', 0.1)  # Hz
        self.aoa_max = kwargs.get('aoa_max', np.pi)
        self.aod_max = kwargs.get('aod_max', np.pi)
        self.tof_max = kwargs.get('tof_max', 200e-9)
        self.dop_max = kwargs.get('dop_max', 20)  # Hz
        self.aoa_cos_search_range = np.arange(-1, 1, self.aoa_cos_search_step)
        self.aod_search_range = np.arange(0, self.aod_max, self.aod_search_step)
        self.tof_search_range = np.arange(0, self.tof_max, self.tof_search_step)
        self.dop_search_range = np.arange(-self.dop_max, self.dop_max, self.dop_search_step)

        self.nn_layers = kwargs.get('nn_layers', 3)
        self.nn_unit_per_layer = kwargs.get('nn_unit_per_layer',
                                            2 * (self.fftsize - 1) * self.rx_antennas * self.tx_antennas)
        self.nn_activation = kwargs.get('nn_activation', 'elu')
        self.nn_loss_func = kwargs.get('nn_loss_func', 'mae')
        self.nn_metrics = kwargs.get('nn_metrics', ['mae'])
        self.nn_optimizer = kwargs.get('nn_optimizer', 'adam')
        self.nn_epochs = kwargs.get('nn_epochs', 1000)
        self.nn_batch_size = kwargs.get('nn_batch_size', 32)
        self.nn_sparse_y = kwargs.get('nn_sparse_y', False)
        self.nn_data_snr_range = kwargs.get('nn_data_snr_range', (10, 20))
        self.max_physical_paths = kwargs.get('max_physical_paths', 5)
        self.nn_validation_split = kwargs.get('nn_validation_split', 0.1)
        self.nn_data_size = kwargs.get('nn_data_size', 10000)
        self.nn_training_data_size = None
        self.nn_validation_data_size = None
        self.nn_working_folder = None
        # self.nn_training_data_filename = None
        # self.nn_validation_data_filename = None
        self.nn_training_data_generator = None
        self.nn_validation_data_generator = None

        self.nn_1d_model_tof = None
        self.nn_2d_model_aoa_tof = None
        # self.m = Mdtrack(**kwargs)

        if self.debug:
            logger.setLevel(logging.DEBUG)

        logger.debug(kwargs)

    def get_aoa_tof_heatmap(self, H) -> np.ndarray:
        assert H.shape == (self.fftsize, self.rx_antennas)
        hmap = np.zeros((len(self.aoa_cos_search_range), len(self.tof_search_range)))
        for i, aoa_cos in enumerate(self.aoa_cos_search_range):
            for j, tof_s in enumerate(self.tof_search_range):
                tof_effect = np.exp(-1j * 2 * np.pi * tof_s * self.sp_light / self.wavelength)
                dist_traveled = self.ant_spacing * np.arange(self.rx_antennas) * aoa_cos
                aoa_effect = np.exp(-1j * 2 * np.pi * dist_traveled / self.wavelength[:, np.newaxis])
                cor = tof_effect[:, np.newaxis] * aoa_effect
                hmap[i, j] = np.abs(np.sum(H * cor.conjugate())) ** 2
        return hmap

    def get_ml_aoa_tof_heatmap(self, H, model_2d):
        assert H.shape == (self.fftsize, self.rx_antennas, self.tx_antennas)
        x = np.array([data_vectorization(data_normalization(H))])
        y = model_2d(x)
        return y[0].numpy()

    def show_aoa_tof_heatmap(self, image, peaks):
        # image = image.T
        pl.figure()
        pl.imshow(image, aspect='auto', cmap=pl.get_cmap('jet'))
        for i, peak  in enumerate(peaks):
            aoa_rad, tof = peak
            print(f"peak{i+1} aoa(deg)={np.rad2deg(aoa_rad):.2f}, tof(ns)={tof*1e9:.2f}")
            pl.annotate(f"x", xy=(
                np.where(self.tof_search_range == tof)[0][0], np.where(np.arccos(self.aoa_cos_search_range) == aoa_rad)[0][0]),
                            fontsize='large', color='w')
        tick_pos = np.arange(0, len(self.tof_search_range), len(self.tof_search_range) // 5)
        pl.xticks(tick_pos, np.round(self.tof_search_range[tick_pos] * 1e9, 2))
        tick_pos = np.arange(0, len(self.aoa_cos_search_range), len(self.aoa_cos_search_range) // 5)
        pl.yticks(tick_pos, np.round(np.rad2deg(np.arccos(self.aoa_cos_search_range[tick_pos])), 2))
        pl.xlabel('ToF (ns)')
        pl.ylabel('AoA (deg)')
        pl.show()

    def detect_peaks(self, image):
        """
        A code snippet from stackoverflow.
        Takes an image and detect the peaks usingthe local maximum filter.
        Returns a boolean mask of the peaks (i.e. 1 when the pixel's value is the neighborhood maximum, 0 otherwise)
        """
        # threshold = 1 / 10
        # window = (5, 5)
        # local_max = maximum_filter(image, size=window, mode='wrap')
        # new_image = image / local_max
        # detected_peaks = np.logical_and(image >= threshold, new_image == 1)

        show_plots = True
        if show_plots:
            pl.figure()
            pl.subplot(2, 2, 1)
            pl.imshow(image, aspect="auto")
            pl.colorbar()

        r, c = image.shape
        win = 5
        y = np.array(image)
        th = ndimage.filters.gaussian_filter(y, win)
        y = np.where(y < th, 0, y)
        if show_plots:
            pl.subplot(2, 2, 2)
            pl.imshow(y, aspect="auto")
            pl.colorbar()

        filt = ndimage.filters.maximum_filter(y, win)
        th = np.mean(y)
        filt = np.where(filt <= th, th, filt)

        y = y / filt

        if show_plots:
            pl.subplot(2, 2, 3)
            pl.imshow(y, aspect="auto")
            pl.colorbar()

        th = 1
        y = np.where(y >= th, 1, 0)

        if show_plots:
            pl.subplot(2, 2, 4)
            pl.imshow(y, aspect="auto")
            ys, xs = np.where(y == 1)
            for i in range(len(ys)):
                pl.annotate(f"x", xy=(xs[i], ys[i]), fontsize='large', color='w')
            tick_pos = np.arange(0, len(self.tof_search_range), len(self.tof_search_range) // 3)
            pl.xticks(tick_pos, np.round(self.tof_search_range[tick_pos] * 1e9, 2))
            tick_pos = np.arange(0, len(self.aoa_cos_search_range), len(self.aoa_cos_search_range) // 5)
            pl.yticks(tick_pos, np.round(np.rad2deg(np.arccos(self.aoa_cos_search_range[tick_pos])), 2))
            pl.xlabel('ToF (ns)')
            pl.ylabel('AoA (deg)')
            pl.colorbar()
            pl.show()

        detected_peaks = np.where(y == 1)

        return detected_peaks

    def get_ranked_peaks(self, image, debug=False):
        threshold = 0.0
        nomalized_image = image / np.max(image)
        peaks = self.detect_peaks(nomalized_image)
        aoa_idxs, tof_idxs = peaks
        ranked_peaks_in_idx = sorted(zip(aoa_idxs, tof_idxs), key=lambda x: nomalized_image[x[0], x[1]], reverse=True)
        ranked_peaks_in_val = [
            (np.arccos(self.aoa_cos_search_range[aoa_idx]), self.tof_search_range[tof_idx])
            for aoa_idx, tof_idx in ranked_peaks_in_idx if nomalized_image[aoa_idx, tof_idx] >= threshold]

        if debug or self.debug:
            debug_ranked_peaks_in_val = [
                (np.rad2deg(np.arccos(self.aoa_cos_search_range[aoa_idx])), 1e9 * self.tof_search_range[tof_idx],
                 nomalized_image[aoa_idx, tof_idx])
                for aoa_idx, tof_idx in ranked_peaks_in_idx if nomalized_image[aoa_idx, tof_idx] >= threshold]
            for aoa_deg, tof_ns, normalized_peak in debug_ranked_peaks_in_val:
                print(f"aoa(deg)={aoa_deg:.2f}, tof(ns)={tof_ns:.2f}, peak_val={normalized_peak:.2f}")
        return np.array(ranked_peaks_in_val)

    def filter_peaks(self, peaks, aoa_rad_th, tof_s_th):
        res = []
        for peak in peaks:
            aoa_rad, tof = peak
            if not res:
                res.append(peak)
            else:
                fail = False
                for x, y in res:
                    if abs(x - aoa_rad) < aoa_rad_th and abs(y - tof) < tof_s_th:
                        fail = True
                        break
                if not fail:
                    res.append(peak)

        return np.array(res)

    def load_training_data(self, fname):
        with open(fname, 'rb') as f:
            self.training_data_x, self.training_data_y = pickle.load(f)

    def save_training_data(self, data, fname):
        assert isinstance(data, tuple)
        self.pickle_save(data, fname)

    def pickle_save(self, data, fname):
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'wb') as f:
            pickle.dump(data, f)

    def pickle_load(self, fname):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        return data

    def dump_parameters(self, fname):
        data = {
            'carrier frequency (GHz)': f"{self.fc / 1e9 :.3f}",
            'bandwidth (MHz)': f"{self.bw / 1e6 :.2f}",
            'fftsize, rx_antennas, tx_antennas': f"({self.fftsize}, {self.rx_antennas}, {self.tx_antennas})",
            'antenna_spacing (meter)': f"{self.ant_spacing:.2f}",
            'aoa_cos_search_range (cos)': f"(-1, 1, {self.aoa_cos_search_step})",
            'aod_search_range (radian)': f"(0, {self.aod_max:.2f}, {self.aod_search_step})",
            'tof_search_range (nanosecond)': f"(0, {self.tof_max * 1e9:.2f}, {self.tof_search_step * 1e9:.2f})",
            'dopplershift_search_range (Hz)': f"({-self.dop_max:.2f}, {self.dop_max:.2f}, {self.dop_search_step:.2f})",
            'nn_layers': f"{self.nn_layers}",
            'nn_unit_per_layer': f"{self.nn_unit_per_layer}",
            'nn_activation': f"{self.nn_activation}",
            'nn_loss_func': f"{self.nn_loss_func}",
            'nn_metrics': f"{self.nn_metrics}",
            'nn_optimizer': f"{self.nn_optimizer}",
            'nn_training_data_size': f"{self.nn_training_data_size}",
            'nn_validation_data_size': f"{self.nn_validation_data_size}",
            'nn_batch_size': f"{self.nn_batch_size}",
            'nn_epochs': f"{self.nn_epochs}",
            'nn_data_snr_range': f"{self.nn_data_snr_range}"
        }
        self.pickle_save(data, fname)

    # def get_tof_effect(self, tof_s) -> np.ndarray:
    #     res = np.exp(-1j * 2 * np.pi * tof_s * self.sp_light / self.wavelength)
    #     assert res.ndim == 1 and res.shape[0] == self.fftsize
    #     return res
    #
    # def get_aoa_effect(self, aoa_cos_val) -> np.ndarray:
    #     """
    #     AoA affects phases across subcarriers and antennas
    #     :rtype: np.array
    #     """
    #     dist_traveled = self.ant_spacing * np.arange(self.rx_antennas) * aoa_cos_val
    #     res = np.exp(-1j * 2 * np.pi * dist_traveled / self.wavelength[:, np.newaxis])
    #     assert res.shape == (self.fftsize, self.rx_antennas)
    #     return res
    #
    # def data_normalization(self, sig_f):
    #     return sig_f[1:, :, :] / np.exp(1j * np.angle(sig_f[1, 0, 0]))
    #
    # def data_vectorization(self, sig_f):
    #     x = sig_f.ravel()
    #     return np.concatenate((np.real(x), np.imag(x)))

    # def get_1d_tof_data_generator(self, num_data, batch_size, resolution, max_path):
    #     n = num_data // batch_size
    #
    #     # generate tofs that are at least (sampling_interval * resolution) apart
    #     choices_for_tof = np.arange(0, self.tof_max, resolution / self.bw)
    #     for i in range(n):
    #         train_x, train_y = [], []
    #         for _ in range(batch_size):
    #             npath = np.random.randint(1, max_path + 1)
    #             tofs = np.random.choice(choices_for_tof, npath, replace=False)
    #             idxs = tofs / self.tof_search_step
    #             idxs += np.random.randint(-idxs.min(), len(self.tof_search_range) - 1 - idxs.max())
    #             idxs = idxs.astype(int)
    #             target_y = np.zeros(len(self.tof_search_range))
    #             target_y[idxs] = np.random.rand(npath)
    #             sigma = resolution / self.bw / self.tof_search_step / 3
    #             train_y.append(ndimage.gaussian_filter1d(target_y, sigma) * np.sqrt(2 * np.pi) * sigma)
    #
    #             sig_f = np.zeros((self.fftsize, 1, 1), dtype=np.complex)
    #             for tof, mag in zip(self.tof_search_range[idxs], target_y[idxs]):
    #                 sig_f[:, 0, 0] += self.get_tof_effect(tof) * mag
    #
    #             # leave out dc carrier and rescale st. sig_f[0,0,0] has zero phase
    #             sig_f_normalized = self.data_normalization(sig_f)
    #             sig_f_vectorized = self.data_vectorization(sig_f_normalized)
    #             train_x.append(sig_f_vectorized)
    #         yield np.array(train_x), np.array(train_y)

    def gen_1d_tof_training_data(self, resolution, force=True):
        """
        Generate and store ground truth path parameters. During training, channel responses and the corresponding
        targer outputs will be generated on the fly by the generator object.
        :param size:
        :param resolution:
        :param fname:
        :return:
        """
        x_training_raw_paths_fname = f"{self.nn_working_folder}/x_training_raw_paths.pckl"
        x_training_fname = f"{self.nn_working_folder}/x_training.pckl"
        y_training_sparse_fname = f"{self.nn_working_folder}/y_training_sparse.pckl"
        y_training_fname = f"{self.nn_working_folder}/y_training.pckl"

        x_validation_raw_paths_fname = f"{self.nn_working_folder}/x_validation_raw_paths.pckl"
        x_validation_fname = f"{self.nn_working_folder}/x_validation.pckl"
        y_validation_sparse_fname = f"{self.nn_working_folder}/y_validation_sparse.pckl"
        y_validation_fname = f"{self.nn_working_folder}/y_validation.pckl"

        cores = os.cpu_count()

        if force or not os.path.exists(x_training_raw_paths_fname):
            args = (self.nn_data_size, resolution, self.bw, self.tof_max, self.max_physical_paths)
            raw_paths = map_multiprocess(cores, gen_1d_tof_physical_paths, *args)
            print('finished raw_paths')
            x_validation_raw_paths = raw_paths[: int(self.nn_validation_split * self.nn_data_size)]
            x_training_raw_paths = raw_paths[int(self.nn_validation_split * self.nn_data_size):]

            self.pickle_save(x_training_raw_paths, x_training_raw_paths_fname)
            self.pickle_save(x_validation_raw_paths, x_validation_raw_paths_fname)
            self.nn_training_data_size = len(x_training_raw_paths)
            self.nn_validation_data_size = len(x_validation_raw_paths)

            # generate channel responses for x_training
            sig_generator_kwargs = {'bw': self.bw, 'fc': self.fc, 'tx_antennas': self.tx_antennas,
                                    'rx_antennas': self.rx_antennas, 'ant_spacing': self.ant_spacing}
            sig_generator = Signal(**sig_generator_kwargs)
            args = (x_training_raw_paths, self.nn_data_snr_range, sig_generator)
            x_train = map_multiprocess(cores, from_path_to_channel, *args)
            print('finished x_train')
            # generate channel responses for x_validation
            args = (x_validation_raw_paths, self.nn_data_snr_range, sig_generator)
            x_valid = map_multiprocess(cores, from_path_to_channel, *args)
            print('finished x_valid')
            self.pickle_save(x_train, x_training_fname)
            self.pickle_save(x_valid, x_validation_fname)

            if self.nn_sparse_y:
                raise NotImplementedError
            else:
                # TODO: memory explosion
                args = (x_training_raw_paths, self.tof_search_range, self.tof_search_step, resolution, self.bw)
                y_train = map_multiprocess(cores, from_path_to_1d_tof_dense_target, *args)
                # y_train = np.array(from_path_to_1d_tof_dense_target(*args))
                print('finished y_train')
                args = (x_validation_raw_paths, self.tof_search_range, self.tof_search_step, resolution, self.bw)
                y_valid = map_multiprocess(cores, from_path_to_1d_tof_dense_target, *args)
                # y_valid = np.array(from_path_to_1d_tof_dense_target(*args))
                print('finished y_valid')

                self.pickle_save(y_train, y_training_fname)
                self.pickle_save(y_valid, y_validation_fname)

    def setup_1d_tof_datagenerator(self, resolution):
        if not self.nn_sparse_y:
            return
        else:
            raise NotImplementedError


    def compile_1d_tof_model(self):
        model = keras.models.Sequential()
        model.add(keras.Input(shape=(2 * (self.fftsize - 1) * self.rx_antennas * self.tx_antennas,)))
        for _ in range(self.nn_layers - 1):
            model.add(keras.layers.Dense(units=self.nn_unit_per_layer, activation=self.nn_activation))
        model.add(keras.layers.Dense(units=len(self.tof_search_range), activation=self.nn_activation))
        model.compile(optimizer=self.nn_optimizer, loss=self.nn_loss_func, metrics=self.nn_metrics)
        self.nn_1d_model_tof = model

    def train_1d_tof_model(self, verbose=1):
        model = self.nn_1d_model_tof
        early_stop_no_gain = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50,
                                                           verbose=1, mode='auto')
        save_best = keras.callbacks.ModelCheckpoint(filepath=f"{self.nn_working_folder}/model.h5", verbose=0,
                                                    monitor='val_loss', save_best_only=True)
        callback_list = [save_best, early_stop_no_gain]
        if self.nn_sparse_y:
            raise NotImplementedError
            # history = model.fit_generator(generator=self.nn_training_data_generator,
            #                               validation_data=self.nn_validation_data_generator,
            #                               epochs=self.nn_epochs,
            #                               use_multiprocessing=True,
            #                               workers=os.cpu_count(),
            #                               verbose=verbose,
            #                               callbacks=callback_list)
        else:
            x_training = self.pickle_load(f"{self.nn_working_folder}/x_training.pckl")
            y_training = self.pickle_load(f"{self.nn_working_folder}/y_training.pckl")
            x_validation = self.pickle_load(f"{self.nn_working_folder}/x_validation.pckl")
            y_validation = self.pickle_load(f"{self.nn_working_folder}/y_validation.pckl")
            history = model.fit(x=x_training,
                                y=y_training,
                                batch_size=self.nn_batch_size,
                                validation_data=(x_validation, y_validation),
                                epochs=self.nn_epochs,
                                verbose=verbose,
                                callbacks=callback_list)

        model.save(f"{self.nn_working_folder}/model.h5")
        self.pickle_save(history.history, f"{self.nn_working_folder}/history.pckl")
        self.dump_parameters(f"{self.nn_working_folder}/parameters.pckl")


    def gen_2d_aoa_tof_training_data(self, resolution, force=True):
        x_training_raw_paths_fname = f"{self.nn_working_folder}/x_training_raw_paths.pckl"
        x_training_fname = f"{self.nn_working_folder}/x_training.pckl"
        y_training_sparse_fname = f"{self.nn_working_folder}/y_training_sparse.pckl"
        y_training_fname = f"{self.nn_working_folder}/y_training.pckl"

        x_validation_raw_paths_fname = f"{self.nn_working_folder}/x_validation_raw_paths.pckl"
        x_validation_fname = f"{self.nn_working_folder}/x_validation.pckl"
        y_validation_sparse_fname = f"{self.nn_working_folder}/y_validation_sparse.pckl"
        y_validation_fname = f"{self.nn_working_folder}/y_validation.pckl"

        cores = os.cpu_count()

        if force or not os.path.exists(x_training_raw_paths_fname):
            args = (self.nn_data_size, resolution, self.bw, self.tof_max, self.rx_antennas, self.max_physical_paths)
            raw_paths = map_multiprocess(cores, gen_2d_aoa_tof_physical_paths, *args)
            print('finished raw_paths')
            x_validation_raw_paths = raw_paths[: int(self.nn_validation_split * self.nn_data_size)]
            x_training_raw_paths = raw_paths[int(self.nn_validation_split * self.nn_data_size):]

            self.pickle_save(x_training_raw_paths, x_training_raw_paths_fname)
            self.pickle_save(x_validation_raw_paths, x_validation_raw_paths_fname)
            self.nn_training_data_size = len(x_training_raw_paths)
            self.nn_validation_data_size = len(x_validation_raw_paths)

            # generate channel responses for x_training
            sig_generator_kwargs = {'bw': self.bw, 'fc': self.fc, 'tx_antennas': self.tx_antennas,
                                    'rx_antennas': self.rx_antennas, 'ant_spacing': self.ant_spacing}
            sig_generator = Signal(**sig_generator_kwargs)
            args = (x_training_raw_paths, self.nn_data_snr_range, sig_generator)
            x_train = map_multiprocess(cores, from_path_to_channel, *args)
            print('finished x_train')
            # generate channel responses for x_validation
            args = (x_validation_raw_paths, self.nn_data_snr_range, sig_generator)
            x_valid = map_multiprocess(cores, from_path_to_channel, *args)
            print('finished x_valid')
            self.pickle_save(x_train, x_training_fname)
            self.pickle_save(x_valid, x_validation_fname)

            if self.nn_sparse_y:
                args = (x_training_raw_paths, self.tof_search_range, self.tof_search_step, self.aoa_cos_search_range,
                        self.aoa_cos_search_step)
                y_train = map_multiprocess(cores, from_path_to_2d_aoa_tof_sparse_target, *args)
                print('finished y_train_sparse')
                args = (x_validation_raw_paths, self.tof_search_range, self.tof_search_step, self.aoa_cos_search_range,
                        self.aoa_cos_search_step)
                y_valid = map_multiprocess(cores, from_path_to_2d_aoa_tof_sparse_target, *args)
                print('finished y_valid_sparse')
                self.pickle_save(y_train, y_training_sparse_fname)
                self.pickle_save(y_valid, y_validation_sparse_fname)

            else:
                # TODO: memory explosion
                args = (x_training_raw_paths, self.tof_search_range, self.tof_search_step, self.aoa_cos_search_range,
                        self.aoa_cos_search_step, resolution, self.bw, self.rx_antennas)
                # y_train = map_multiprocess(cores, from_path_to_2d_dense_target, *args)
                y_train = np.array(from_path_to_2d_aoa_tof_dense_target(*args))
                print('finished y_train')
                args = (x_validation_raw_paths, self.tof_search_range, self.tof_search_step, self.aoa_cos_search_range,
                        self.aoa_cos_search_step, resolution, self.bw, self.rx_antennas)
                # y_valid = map_multiprocess(cores, from_path_to_2d_dense_target, *args)
                y_valid = np.array(from_path_to_2d_aoa_tof_dense_target(*args))
                print('finished y_valid')

                self.pickle_save(y_train, y_training_fname)
                self.pickle_save(y_valid, y_validation_fname)

    def setup_2d_aoa_tof_datagenerator(self, resolution):
        if not self.nn_sparse_y:
            return

        # raw physical path parameters
        x_training = self.pickle_load(f"{self.nn_working_folder}/x_training.pckl")
        y_training_sparse = self.pickle_load(f"{self.nn_working_folder}/y_training_sparse.pckl")
        x_validation = self.pickle_load(f"{self.nn_working_folder}/x_validation.pckl")
        y_validation_sparse = self.pickle_load(f"{self.nn_working_folder}/y_validation_sparse.pckl")

        data_generator_kwargs = {'bw': self.bw, 'rx_antennas': self.rx_antennas,
                                 'tof_search_range': self.tof_search_range, 'tof_search_step': self.tof_search_step,
                                 'aoa_cos_search_range': self.aoa_cos_search_range,
                                 'aoa_cos_search_step': self.aoa_cos_search_step}

        self.nn_training_data_generator = Generator_2d_aoa_tof(x=x_training, y_sparse=y_training_sparse,
                                                               batch_size=self.nn_batch_size, resolution=resolution,
                                                               **data_generator_kwargs)

        self.nn_validation_data_generator = Generator_2d_aoa_tof(x=x_validation, y_sparse=y_validation_sparse,
                                                                 batch_size=self.nn_batch_size, resolution=resolution,
                                                                 **data_generator_kwargs)

    def compile_2d_aoa_tof_model(self):
        model = keras.models.Sequential()
        model.add(keras.Input(shape=(2 * (self.fftsize - 1) * self.rx_antennas * self.tx_antennas,)))
        for _ in range(self.nn_layers - 1):
            model.add(keras.layers.Dense(units=self.nn_unit_per_layer, activation=self.nn_activation))
        model.add(keras.layers.Dense(units=len(self.tof_search_range) * len(self.aoa_cos_search_range),
                                     activation=self.nn_activation))
        model.add(keras.layers.Reshape((len(self.aoa_cos_search_range), len(self.tof_search_range))))
        model.compile(optimizer=self.nn_optimizer, loss=self.nn_loss_func, metrics=self.nn_metrics)
        self.nn_2d_model_aoa_tof = model

    def train_2d_aoa_tof_model(self, verbose=1):
        model = self.nn_2d_model_aoa_tof
        early_stop_no_gain = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=200,
                                                           verbose=1, mode='auto')
        save_best = keras.callbacks.ModelCheckpoint(filepath=f"{self.nn_working_folder}/model.h5", verbose=0,
                                                    monitor='val_loss', save_best_only=True)
        callback_list = [save_best, early_stop_no_gain]
        if self.nn_sparse_y:
            history = model.fit_generator(generator=self.nn_training_data_generator,
                                          validation_data=self.nn_validation_data_generator,
                                          epochs=self.nn_epochs,
                                          use_multiprocessing=True,
                                          workers=os.cpu_count(),
                                          verbose=verbose,
                                          callbacks=callback_list)
        else:
            x_training = self.pickle_load(f"{self.nn_working_folder}/x_training.pckl")
            y_training = self.pickle_load(f"{self.nn_working_folder}/y_training.pckl")
            x_validation = self.pickle_load(f"{self.nn_working_folder}/x_validation.pckl")
            y_validation = self.pickle_load(f"{self.nn_working_folder}/y_validation.pckl")
            history = model.fit(x=x_training,
                                y=y_training,
                                batch_size=self.nn_batch_size,
                                validation_data=(x_validation, y_validation),
                                epochs=self.nn_epochs,
                                verbose=verbose,
                                callbacks=callback_list)

        model.save(f"{self.nn_working_folder}/model.h5")
        self.pickle_save(history.history, f"{self.nn_working_folder}/history.pckl")
        self.dump_parameters(f"{self.nn_working_folder}/parameters.pckl")


if __name__ == '__main__':
    pass
# a = Optml2()
# m = Mdtrack()
# x = a.gen_training_data(m, number=5)

# model = keras.models.Sequential()
# model.add(keras.Input(shape=(2 * m.fftsize * m.rx_antennas * m.tx_antennas,)))
# model.add(keras.layers.Reshape((m.fftsize * 2, m.rx_antennas)))
# model.add(keras.layers.Dense(m.fftsize * 2, activation='relu'))
# y = a.nn_1d_model_tof(x)
# print(x.shape, y.shape)
