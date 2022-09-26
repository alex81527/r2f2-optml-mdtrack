import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.optimize import minimize
from scipy import ndimage
from mystic.solvers import fmin_powell
from scipy.special import diric
# from ipopt import setLoggingLevel
import logging
import json
import collections
import bisect
from ieee80211.preamble import Preamble
from functools import lru_cache
from mdtrack_mobicom19.core import Parameter
from utils.common import timer
import matplotlib.pyplot as pl

logger = logging.getLogger(__name__)


class R2f2(object):
    """
    Eliminating Channel Feedback in Next-Generation. Cellular Networks. SIGCOMM'16
    """

    def __init__(self, *args, **kwargs):
        # setLoggingLevel(logging.ERROR)
        self.debug = kwargs.get('debug', False)
        self.ltf_f = kwargs.get('ltf_f', Preamble.LEGACY_LTF_F_64)
        self.ltf_t = np.fft.ifft(self.ltf_f)
        self.stf_f = kwargs.get('stf_f', Preamble.LEGACY_STF_F_64)
        self.stf_t = kwargs.get('stf_t', Preamble.LEGACY_STF_T_16)
        self.bw = kwargs.get('bw', 20e6)
        self.fc = kwargs.get('fc', 2.412e9)
        self.sp_light = 3e8
        self.tx_antennas = kwargs.get('tx_antennas', 1)
        self.rx_antennas = kwargs.get('rx_antennas', 3)
        self.ant_spacing = kwargs.get('ant_spacing', self.sp_light / self.fc / 2)
        self.fftsize = len(self.ltf_f)
        self.wavelength = self.sp_light / (np.fft.fftfreq(self.fftsize) * self.bw + self.fc)
        self.tx_aperture = self.tx_antennas * self.ant_spacing
        self.rx_aperture = self.rx_antennas * self.ant_spacing
        self.aoa_search_step = kwargs.get('aoa_search_step', 0.02)  # radian
        self.aod_search_step = kwargs.get('aod_search_step', 0.02)  # radian
        self.tof_search_step = kwargs.get('tof_search_step', 0.5e-9)  # second
        self.dop_search_step = kwargs.get('dop_search_step', 0.1)  # Hz
        self.aoa_max = kwargs.get('aoa_max', np.pi)
        self.aod_max = kwargs.get('aod_max', np.pi)
        self.tof_max = kwargs.get('tof_max', 200e-9)
        self.dop_max = kwargs.get('dop_max', 20)  # Hz
        self.aoa_search_range = np.arange(0, self.aoa_max, self.aoa_search_step)
        self.aod_search_range = np.arange(0, self.aod_max, self.aod_search_step)
        self.tof_search_range = np.arange(0, self.tof_max, self.tof_search_step)
        self.dop_search_range = np.arange(-self.dop_max, self.dop_max, self.dop_search_step)
        self.channel_est = None
        self.stopping_criterion_threshold = 0.01 * self.fftsize * self.rx_antennas * self.tx_antennas
        self.stopping_criterion_trivial_gain_threshold = 0.15

        # precompute matrices for better performance
        self.rx_F = self.precompute_matrix_F(self.rx_antennas, self.rx_aperture)  # shape (fftsize, rx_ant, rx_ant)
        self.tx_F = self.precompute_matrix_F(self.tx_antennas, self.tx_aperture)  # shape (fftsize, tx_ant, tx_ant)
        self.rx_F_inv = self.precompute_matrix_F_inv(self.rx_antennas, self.rx_F)
        self.tx_F_inv = self.precompute_matrix_F_inv(self.tx_antennas, self.tx_F)
        # shape (fftsize, rx_ant, aoa_search_range)
        self.rx_S = self.precompute_matrix_S(self.rx_antennas, self.rx_aperture, self.aoa_search_range)
        # shape (fftsize, tx_ant, aod_search_range)
        self.tx_S = self.precompute_matrix_S(self.tx_antennas, self.tx_aperture, self.aod_search_range)
        self.D = self.precompute_matrix_D()  # shape (fftsize, tof_search_range)

        self.aoa_search_mat = self.precompute_aoa_search_mat()
        self.aod_search_mat = self.precompute_aod_search_mat()
        self.tof_search_vec = self.precompute_tof_search_vec_in_freqdomain()
        # self.dop_search_mat = self.precompute_dop_search_mat()

        if self.debug:
            logger.setLevel(logging.DEBUG)
        logger.debug(kwargs)

    def precompute_matrix_S(self, antennas, aperture, angle_search_range):
        """
        Angles (AoA/AoD) observed across antennas are convolved with a sinc functions of width corresponding to the
        wavelength. Once the number of observing antennas is decided, we know how many points from a sinc function
        need to be computed. To minimize on-the-fly computation, center of the sinc is placed at discrete locations
        specified by angle_search_range which is the search step our solver will take.
        Refer to eq. (9) in the paper.
        :return: matrix of shape (fftsize, antennas, angle_search_range)
        """
        res = np.zeros((self.fftsize, antennas, len(angle_search_range)), dtype=np.complex)
        # for i in range(self.fftsize):
        #     for j, angle in enumerate(angle_search_range):
        #         psi_prime = self.wavelength[i] / aperture
        #         psi_s = np.arange(antennas) * psi_prime
        #         psi_j = np.cos(angle)
        #         res[i, :, j] = aperture / self.wavelength[i] * np.sinc(aperture / self.wavelength[i] * (psi_s - psi_j))

        for i in range(self.fftsize):
            for j, angle_rad in enumerate(angle_search_range):
                center = 2 * np.pi * self.ant_spacing * np.cos(angle_rad) / self.wavelength[i]
                sampled_cos_pos = 2 * np.pi * self.ant_spacing * np.linspace(-1, 1, antennas, endpoint=False) / \
                                  self.wavelength[i]
                res[i, :, j] = self.dirichlet_kernel(sampled_cos_pos, antennas, center)

        return res

    def precompute_matrix_D(self):
        """
        Refer to eq. (12) in the paper. To be computationally efficient, only the diagonal values needs to be stored.
        There's no need to actually instantiate a diagonal matrix.
        :return: Diagonals of the matrix D of shape (fftsize, tof_search_range)
        """
        res = np.exp(-1j * 2 * np.pi * self.tof_search_range * self.sp_light / self.wavelength[:, np.newaxis])
        assert res.shape == (self.fftsize, len(self.tof_search_range))
        return res

    def precompute_matrix_F(self, antennas, aperture):
        """
        Standard NxN fourier matrix which converts angles (AoA/AoD) into freq response across antennas.
        Refer to eq. (9) in the paper.
        :return: Fourier matrix of shape (fftsize, antennas, angles)
        """
        res = np.zeros((self.fftsize, antennas, antennas), dtype=np.complex)
        # idxs = np.arange(antennas)[:, np.newaxis] * np.arange(antennas)
        # for i, wav_len in enumerate(self.wavelength):
        #     psi_prime = wav_len / aperture
        #     res[i, :, :] = np.exp(-1j * 2 * np.pi * self.ant_spacing * psi_prime / wav_len * idxs)

        cos_vals = np.linspace(-1, 1, antennas, endpoint=False)
        for i, wav_len in enumerate(self.wavelength):
            # array response at antenna k = exp(-1j * 2 * pi * k * ant_spacing * cos(theta) / wav_len)
            # cos(theta) are sampled at equally spaced points between -1 and 1
            for k in range(antennas):
                res[i, k, :] = np.exp(-1j * 2 * np.pi * k * self.ant_spacing * cos_vals / wav_len)

        return res

    def precompute_matrix_F_inv(self, antennas, matrix_F):
        """
        Current implementation does not assume unitary fourier matrix.
        :return: Fourier matrix of shape (fftsize, antennas, angles)
        """
        assert matrix_F.shape == (self.fftsize, antennas, antennas)
        res = matrix_F.copy()
        for i in range(self.fftsize):
            res[i, :, :] = res[i, :, :].T.conjugate() / antennas
        return res

    def precompute_aoa_search_mat(self) -> list:
        aoa_search_mat = [self.get_aoa_effect(aoa_rad) for aoa_rad in self.aoa_search_range]
        assert aoa_search_mat[0].shape == (self.fftsize, self.rx_antennas)
        return aoa_search_mat

    def precompute_aod_search_mat(self) -> list:
        aod_search_mat = [self.get_aod_effect(aod_rad) for aod_rad in self.aod_search_range]
        assert aod_search_mat[0].shape == (self.fftsize, self.tx_antennas)
        return aod_search_mat

    def precompute_tof_search_vec_in_freqdomain(self) -> list:
        tof_search_vec = [self.get_tof_effect(tof) for tof in self.tof_search_range]
        assert tof_search_vec[0].ndim == 1 and tof_search_vec[0].shape[0] == self.fftsize
        return tof_search_vec

    # def precompute_dop_search_mat(self) -> list:
    #     """
    #     This saerch matrix is meant to work with preamble train.
    #     :return:
    #     """
    #     dop_search_mat = []
    #     preamble_len = self.preamble_train.shape[1]
    #     for dop in self.dop_search_range:
    #         dop_mat = []
    #         for i in range(self.preamble_train.shape[0]):
    #             idx_start = i * self.preamble_interval_sample
    #             dop_mat.append(np.exp(1j * 2 * np.pi * dop / self.bw * np.arange(idx_start, idx_start + preamble_len)))
    #         dop_search_mat.append(np.array(dop_mat))
    #     assert dop_search_mat[0].shape == self.preamble_train.shape
    #     return dop_search_mat

    def get_aoa_effect(self, aoa_rad) -> np.ndarray:
        """
        AoA affects phases across subcarriers and antennas
        :rtype: np.array
        """
        dist_traveled = self.ant_spacing * np.arange(self.rx_antennas) * np.cos(aoa_rad)
        res = np.exp(-1j * 2 * np.pi * dist_traveled / self.wavelength[:, np.newaxis])
        assert res.shape == (self.fftsize, self.rx_antennas)
        return res

    def get_aod_effect(self, aod_rad) -> np.ndarray:
        """
        AoD affects phases across subcarriers and antennas
        :rtype: np.array
        """
        dist_traveled = self.ant_spacing * np.arange(self.tx_antennas) * np.cos(aod_rad)
        res = np.exp(-1j * 2 * np.pi * dist_traveled / self.wavelength[:, np.newaxis])
        assert res.shape == (self.fftsize, self.tx_antennas)
        return res

    def get_tof_effect(self, tof_s) -> np.ndarray:
        res = np.exp(-1j * 2 * np.pi * tof_s * self.sp_light / self.wavelength)
        assert res.ndim == 1 and res.shape[0] == self.fftsize
        return res

    def get_aoa_tof_heatmap(self, H) -> np.ndarray:
        assert H.shape == (self.fftsize, self.rx_antennas)
        hmap = np.zeros((len(self.aoa_search_range), len(self.tof_search_range)))
        for i, aoa_rad in enumerate(self.aoa_search_range):
            for j, tof_s in enumerate(self.tof_search_range):
                cor = self.get_tof_effect(tof_s)[:, np.newaxis] * self.get_aoa_effect(aoa_rad)
                hmap[i, j] = np.abs(np.sum(H * cor.conjugate())) ** 2
        return hmap

    def show_aoa_tof_heatmap(self, image, peaks):
        pl.figure()
        pl.imshow(image, aspect='auto', cmap=pl.get_cmap('jet'))
        for i, peak  in enumerate(peaks):
            aoa_rad, tof = peak
            print(f"peak{i+1} aoa(deg)={np.rad2deg(aoa_rad):.2f}, tof(ns)={tof*1e9:.2f}")
            pl.annotate(f"x", xy=(
                np.where(self.tof_search_range == tof)[0][0], np.where(self.aoa_search_range == aoa_rad)[0][0]),
                            fontsize='large', color='w')
        tick_pos = np.arange(0, len(self.tof_search_range), len(self.tof_search_range) // 5)
        pl.xticks(tick_pos, np.round(self.tof_search_range[tick_pos] * 1e9, 2))
        tick_pos = np.arange(0, len(self.aoa_search_range), len(self.aoa_search_range) // 5)
        pl.yticks(tick_pos, np.round(np.rad2deg(self.aoa_search_range[tick_pos]), 2))
        pl.xlabel('ToF (ns)')
        pl.ylabel('AoA (deg)')
        pl.show()

    def show_aoa_tof_cost_func(self):
        hmap = np.zeros((len(self.aoa_search_range), len(self.tof_search_range)))
        for i, aoa_rad in enumerate(self.aoa_search_range):
            for j, tof_s in enumerate(self.tof_search_range):
                hmap[i, j] = self.objective_func(np.array([aoa_rad, tof_s]), [1e5, 1e5])

        pl.figure()
        pl.imshow(hmap, aspect='auto', cmap=pl.get_cmap('jet'))
        tick_pos = np.arange(0, len(self.tof_search_range), len(self.tof_search_range) // 5)
        pl.xticks(tick_pos, np.round(self.tof_search_range[tick_pos] * 1e9, 2))
        tick_pos = np.arange(0, len(self.aoa_search_range), len(self.aoa_search_range) // 5)
        pl.yticks(tick_pos, np.round(np.rad2deg(self.aoa_search_range[tick_pos]), 2))
        pl.xlabel('ToF (ns)')
        pl.ylabel('AoA (deg)')
        pl.colorbar()
        pl.show()

    def dirichlet_kernel(self, w, window_size, center=0):
        return np.exp(1j * (w - center) * (window_size - 1) / 2) * diric(w - center, window_size)

    def get_matrix_S(self, antennas, aperture, cos_vals):
        """
        :return: matrix of shape (fftsize, antennas, angle_search_range)
        """
        res = np.zeros((self.fftsize, antennas, len(cos_vals)), dtype=np.complex)
        for i in range(self.fftsize):
            for j, cos_val in enumerate(cos_vals):
                center = 2 * np.pi * self.ant_spacing * cos_val / self.wavelength[i]
                sampled_cos_pos = 2 * np.pi * self.ant_spacing * np.linspace(-1, 1, antennas, endpoint=False) / \
                                  self.wavelength[i]
                res[i, :, j] = self.dirichlet_kernel(sampled_cos_pos, antennas, center)
        return res

    def get_matrix_D(self, tofs):
        """
        Refer to eq. (12) in the paper. To be computationally efficient, only the diagonal values needs to be stored.
        There's no need to actually instantiate a diagonal matrix.
        :return: Diagonals of the matrix D of shape (fftsize, tof_search_range)
        """
        res = np.exp(-1j * 2 * np.pi * tofs * self.sp_light / self.wavelength[:, np.newaxis])
        assert res.shape == (self.fftsize, len(tofs))
        return res

    def initial_guess(self, channel_est):
        # TODO: for now channel_est is of shape (fftsize, rx_ant, 1)
        assert channel_est.shape[2] == 1
        heatmap = self.get_aoa_tof_heatmap(channel_est[:,:,0])
        peaks = self.filter_peaks(self.get_ranked_peaks(heatmap, debug=True))
        return peaks


    def print_pretty(self, v_list) -> str:
        assert isinstance(v_list, np.ndarray), "Input must be np.ndarray."
        aoa_rads = np.real([Parameter.get_val(x) for x in v_list[:, 0]])
        aod_rads = np.real([Parameter.get_val(x) for x in v_list[:, 1]])
        tofs = np.real([Parameter.get_val(x) for x in v_list[:, 2]])
        dops = np.real([Parameter.get_val(x) for x in v_list[:, 3]])
        gains = [Parameter.get_val(x) for x in v_list[:, 4]]
        return json.dumps(
            {'AOA (deg) ': ','.join([f"{aoa_rad / np.pi * 180: >7.2f}" for aoa_rad in aoa_rads]),
             'AOD (deg) ': ','.join([f"{aod_rad / np.pi * 180: >7.2f}" for aod_rad in aod_rads]),
             'TOF (ns)  ': ','.join([f"{tof_s * 1e9: >7.2f}" for tof_s in tofs]),
             'DOP (Hz)  ': ','.join([f"{dop_hz: >7.2f}" for dop_hz in dops]),
             'Gain (Mag)': ','.join([f"{np.abs(gain): >7.2f}" for gain in gains]),
             'Gain (rad)': ','.join([f"{np.angle(gain): >7.2f}" for gain in gains])},
            indent=4)

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
            ys, xs = np.where(y==1)
            for i in range(len(ys)):
                pl.annotate(f"x", xy=(xs[i], ys[i]), fontsize='large', color='w')
            tick_pos = np.arange(0, len(self.tof_search_range), len(self.tof_search_range) // 3)
            pl.xticks(tick_pos, np.round(self.tof_search_range[tick_pos] * 1e9, 2))
            tick_pos = np.arange(0, len(self.aoa_search_range), len(self.aoa_search_range) // 5)
            pl.yticks(tick_pos, np.round(np.rad2deg(self.aoa_search_range[tick_pos]), 2))
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
            (self.aoa_search_range[aoa_idx], self.tof_search_range[tof_idx])
            for aoa_idx, tof_idx in ranked_peaks_in_idx if nomalized_image[aoa_idx, tof_idx] >= threshold]

        if debug or self.debug:
            debug_ranked_peaks_in_val = [
                (np.rad2deg(self.aoa_search_range[aoa_idx]), 1e9 * self.tof_search_range[tof_idx],
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

    def convert_val_to_idx(self, range_array, val_list):
        return [bisect.bisect_left(range_array, x) if x <= range_array[-1] else len(range_array) - 1 for x in val_list]

    def objective_func(self, x, sol):
        """
        eq. (14)
        :param v_list: of shape (n_paths, 5), where each path is expressed by (AoA, AoD, ToF, Doppler, Complex gain)
        :param channel_est:
        :return:
        """
        # assert isinstance(v_list, np.ndarray)
        assert self.channel_est.shape == (self.fftsize, self.rx_antennas, self.tx_antennas)


        # TODO: consider only AoA for now
        # x_in_integers = list(map(int, x))
        aoa_idxs = self.convert_val_to_idx(self.aoa_search_range, x[::2])
        tof_idxs = self.convert_val_to_idx(self.tof_search_range, x[1::2])
        # logger.debug(f"actual lookup value aoa_deg={np.round(np.rad2deg(self.aoa_search_range[aoa_idxs]), 3)}, "\
        #             f"tof_ns={np.round(self.tof_search_range[tof_idxs] * 1e9, 3)}")
        # eq. (12)
        # (fftsize, rx_ant, n_paths) * (fftsize, 1, n_paths)
        # e.g. fftsize=64, rx_ant=4, n_paths=2, the outcome will be of shape (64, 4, 2)
        SD = self.rx_S[:, :, aoa_idxs] * self.D[:, np.newaxis, tof_idxs]
        SD = SD.reshape(self.fftsize * self.rx_antennas, len(tof_idxs))
        SD_dag = np.linalg.pinv(SD)

        P = np.array([self.rx_F_inv[i, :, :] @ self.channel_est[i, :, 0] for i in range(self.fftsize)])
        P = P.ravel()  # 1D array
        # return np.linalg.norm(P - S @ S_dag @ P) ** 2
        val = np.linalg.norm(P - np.dot(np.dot(SD, SD_dag), P)) ** 2
        # logger.debug(f"aoa_deg={np.round(np.rad2deg(x[::2]), 2)}, tof_ns={np.round(x[1::2] * 1e9, 2)}, " \
        #              f"obj_func_val={val:.2f}")
        if val < sol[1]:
            sol[0] = x
            sol[1] = val


        '''
        aoa_rads = x[::2]
        tof_secs = x[1::2]
        S = self.get_matrix_S(self.rx_antennas, self.rx_aperture, np.cos(aoa_rads))  # (fftsize, rx_ant, npaths)
        D = self.get_matrix_D(tof_secs)  # (fftsize, npaths)
        SD = S * D[:, np.newaxis, :]
        SD = SD.reshape(self.fftsize * self.rx_antennas, len(tof_secs))
        SD_dag = np.linalg.pinv(SD)
        P = np.array([self.rx_F_inv[i, :, :] @ self.channel_est[i, :, 0] for i in range(self.fftsize)])
        P = P.ravel()  # 1D array
        # val = np.linalg.norm(P - SD @ SD_dag @ P) ** 2
        val = np.linalg.norm(P - np.dot(np.dot(SD, SD_dag), P)) ** 2
        logger.debug(f"aoa_deg={np.round(np.rad2deg(aoa_rads), 2)}, tof_ns={np.round(tof_secs * 1e9, 2)}, " \
                     f"obj_func_val={val:.2f}")
        if val<sol[1]:
            sol[0] = x
            sol[1] = val
        '''

        return val

    def solve_optimization(self, init_guess):
        """
        Solves the optimization problem in Sec. 5.3 eq. (15)
        :param init_guess: list of tuples (aoa_idx, tof_idx)
        :return: OptimizeResult returned by scipy.optimize.minimize
        """
        # cos ranges from -1 to 1,
        bnds = [(0, self.aoa_max), (0, self.tof_max)] * (len(init_guess) // 2)
        # it does not return the minimum achieved
        # TODO: a quick workaround, pass a mutable into the objective function to keep track of the min
        sol = [init_guess, 1e9]
        res = fmin_powell(self.objective_func, init_guess, bounds=bnds, args=(sol,), disp=0)
        # res = minimize(self.objective_func, init_guess, args=(sol,), method='Powell', bounds=bnds)
        # res = minimize(self.objective_func, init_guess, args=(sol,), method='TNC', bounds=bnds)
        # res = minimize(self.objective_func, init_guess, args=(sol,), method='SLSQP', bounds=bnds)

        # mimic scipy's OptimizeResult
        nt = collections.namedtuple('Sol', ['x', 'fun', 'res'])
        return nt(sol[0], sol[1], res)
        # return minimize(self.objective_func, init_guess, method='SLSQP', bounds=bnds, options={'disp': True})
        # return minimize(self.objective_func, init_guess, method='L-BFGS-B', bounds=bnds, options={'disp': True})

        # TODO: setup the problem for ipopt
        # return minimize_ipopt(self.objective_func, init_guess, bounds=bnds)

    def conditioning(self, best_res, aoa_rad_th, tof_s_th):
        res = []
        aoa_rads = best_res.x[::2]
        tof_secs = best_res.x[1::2]
        for aoa_rad, tof in zip(aoa_rads, tof_secs):
            if not res:
                res.append(aoa_rad)
                res.append(tof)
            else:
                fail = False
                for i in range(len(res)//2):
                    x, y = res[2*i], res[2*i+1]
                    if abs(x - aoa_rad) < aoa_rad_th:# and abs(y - tof) < tof_s_th:
                        fail = True
                        break
                if not fail:
                    res.append(aoa_rad)
                    res.append(tof)

        nt = collections.namedtuple('Sol', ['x'])
        return nt(np.array(res))

    def stopping_criterion_below_threshold(self, cur_res):
        """
        Return true if the value of objective funciton is lower than a predefined threshold.
        :param cur_res: OptimizeResult returned by scipy.optimize.minimize
        :param last_res: OptimizeResult returned by scipy.optimize.minimize
        :return:
        """
        return False

    def stopping_criterion_trivial_decrease(self, cur_res, last_res):
        """
        Return true if the value of objective funciton has trivial decrease in this iteration.
        :param cur_res: OptimizeResult returned by scipy.optimize.minimize
        :param last_res: OptimizeResult returned by scipy.optimize.minimize
        :return:
        """
        return False

    def get_physical_path_parameters(self, res):
        aoa_rads = res.x[::2]
        tof_secs = res.x[1::2]
        aoa_idxs = self.convert_val_to_idx(self.aoa_search_range, res.x[::2])
        tof_idxs = self.convert_val_to_idx(self.tof_search_range, res.x[1::2])

        # eq. (12)
        # (fftsize, rx_ant, n_paths) * (fftsize, 1, n_paths)
        # e.g. fftsize=64, rx_ant=4, n_paths=2, the outcome will be of shape (64, 4, 2)
        S = self.get_matrix_S(self.rx_antennas, self.rx_aperture, np.cos(aoa_rads))  # (fftsize, rx_ant, npaths)
        D = self.get_matrix_D(tof_secs)  # (fftsize, npaths)
        SD = S * D[:, np.newaxis, :]
        S = SD

        # S = self.rx_S[:, :, aoa_idxs] * self.D[:, np.newaxis, tof_idxs]
        S = S.reshape(self.fftsize * self.rx_antennas, len(tof_idxs))
        # S_dag = np.linalg.pinv(S)

        P = np.array([self.rx_F_inv[i, :, :] @ self.channel_est[i, :] for i in range(self.fftsize)])
        P = P.ravel()  # 1D array

        complex_gains, residuals, rank, singular_val = np.linalg.lstsq(S, P, rcond=None)
        # complex_gains = S_dag @ P
        v_list = [np.array([aoa_rad, 0.0, tof_sec, 0.0, a]) for aoa_rad, tof_sec, a in
                  zip(res.x[::2], res.x[1::2], complex_gains)]
        return np.array(v_list)

    def resolve_multipath(self, sig_t):
        """
        Sec. 5.3 in the paper.
        :param sig_t: time domain ltf samples of shape (fftsize, rx_ant, tx_ant)
        :return:
        """
        assert sig_t.shape == (self.fftsize, self.rx_antennas, self.tx_antennas)
        self.channel_est = np.fft.fft(sig_t, axis=0) * self.ltf_f[:, np.newaxis, np.newaxis]
        # ranked_peaks = self.initial_guess(self.channel_est)
        heatmap = self.get_aoa_tof_heatmap(self.channel_est[:, :, 0])
        ranked_peaks = self.get_ranked_peaks(heatmap, debug=True)

        aoa_rad_th, tof_s_th = 2/self.rx_antennas/2, 1/self.bw/2
        filtered_peaks = self.filter_peaks(ranked_peaks, aoa_rad_th, tof_s_th)

        for i, peak in enumerate(filtered_peaks): # np.array of pairs
            init_guess = ranked_peaks[:i+1].ravel()
            res = self.solve_optimization(init_guess)
            if i == 0:
                best_res = res
            elif res.fun > best_res.fun * (1.0 - self.stopping_criterion_trivial_gain_threshold):
                logger.debug(f"Trivial decrease in objective functiton {100*(res.fun-best_res.fun)/best_res.fun:.2} %")
                break
            elif res.fun < self.stopping_criterion_threshold:
                logger.debug(f"Objective function value {res.fun:.3f} below stopping threshold {self.stopping_criterion_threshold:.2f}")
                best_res = res
                break
            elif res.fun < best_res.fun:
                best_res = res


            pars = self.get_physical_path_parameters(res)
            logger.debug(f"N={i + 1} paths, obj_func_val={res.fun:.3f}, {self.print_pretty(pars)}")

        # remove paths that are too close by
        best_res = self.conditioning(best_res, aoa_rad_th, tof_s_th)
        return self.get_physical_path_parameters(best_res)

    def resolve_multipath_ml(self, sig_t, p, model_2d):
        """
        Sec. 5.3 in the paper.
        :param sig_t: time domain ltf samples of shape (fftsize, rx_ant, tx_ant)
        :return:
        """
        assert sig_t.shape == (self.fftsize, self.rx_antennas, self.tx_antennas)
        self.channel_est = np.fft.fft(sig_t, axis=0) * self.ltf_f[:, np.newaxis, np.newaxis]
        heatmap = p.get_ml_aoa_tof_heatmap(self.channel_est, model_2d)
        ranked_peaks = p.get_ranked_peaks(heatmap, debug=True)

        aoa_rad_th, tof_s_th = 2 / self.rx_antennas / 2, 1 / self.bw / 2
        filtered_peaks = p.filter_peaks(ranked_peaks, aoa_rad_th, tof_s_th)
        best_res = self.solve_optimization(filtered_peaks.ravel())
        best_res = self.conditioning(best_res, aoa_rad_th, tof_s_th)
        return self.get_physical_path_parameters(best_res)


        # for i, peak in enumerate(ranked_peaks): # np.array of pairs
        #     init_guess = ranked_peaks[:i+1].ravel()
        #     res = self.solve_optimization(init_guess)
        #     if i == 0:
        #         best_res = res
        #
        #     if res.fun > best_res.fun * (1.0 - self.stopping_criterion_trivial_gain_threshold):
        #         logger.debug(f"Trivial decrease in objective functiton {100*(res.fun-best_res.fun)/best_res.fun:.2} %")
        #         break
        #     elif res.fun < self.stopping_criterion_threshold:
        #         logger.debug(f"Objective function value {res.fun:.3f} below stopping threshold {self.stopping_criterion_threshold:.2f}")
        #         best_res = res
        #         break
        #     elif res.fun < best_res.fun:
        #         best_res = res
        #
        #
        #     pars = self.get_physical_path_parameters(res)
        #     logger.debug(f"N={i + 1} paths, obj_func_val={res.fun:.3f}, {self.print_pretty(pars)}")
        #
        # # remove paths that are too close by
        # best_res = self.conditioning(best_res)
        # return self.get_physical_path_parameters(best_res)