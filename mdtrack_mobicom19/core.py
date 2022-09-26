import numpy as np
import bisect
import matplotlib.pyplot as pl
import time
import json
from utils.common import Db, Snr
from ieee80211.preamble import Preamble
import logging

logger = logging.getLogger(__name__)


class Parameter(object):
    def __init__(self, val=np.complex(0, 0), idx=0):
        self.val = val
        self.idx = idx

    def set_val_and_idx(self, val, idx):
        # assert (isinstance(val, np.complex))
        # assert isinstance(idx, int)
        self.val = val
        self.idx = idx

    @staticmethod
    def get_val(x):
        return x.val if isinstance(x, Parameter) else x

    def __sub__(self, other):
        return self.idx - other.idx


class Mdtrack(object):
    """
    mD-Track: Leveraging Multi-Dimensionality in Passive Indoor Wi-Fi Tracking. MobiCom'19
    """

    def __init__(self, *args, **kwargs):
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
        # section 6.3, default step sizes used in the paper
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
        self.debug = kwargs.get('debug', False)
        self.per_tone_noise_mw = kwargs.get('per_tone_noise_mw', 1e-5)
        self.rf_chain_noise_pwr = kwargs.get('rf_chain_noise_pwr', [1e-5] * self.rx_antennas)
        self.initial_est_stop_threshold_db = kwargs.get('initial_est_stop_threshold_db', 1.0)
        self.preamble_interval_sec = kwargs.get('preamble_interval_sec', 25e-3)  # in second
        self.preamble_repeat_cnt = kwargs.get('preamble_repeat_cnt', 40)
        self.preamble_interval_sample = int(self.preamble_interval_sec * self.bw)
        # TODO: hardcode preamble for now
        # self.preamble_train = np.array([np.tile(self.ltf_t, 2) for _ in range(self.preamble_repeat_cnt)])
        self.preamble_train = np.array([self.ltf_t for _ in range(self.preamble_repeat_cnt)])
        # assert self.preamble_train.shape == (self.preamble_repeat_cnt, len(self.ltf_t))

        # precompute matrices which will be used in the future
        self.aoa_search_mat = self.precompute_aoa_search_mat()  # shape (aoas, preambles, fftsize, rx_ant)
        self.aod_search_mat = self.precompute_aod_search_mat()  # shape (aods, preambles, fftsize, tx_ant)
        self.tof_search_mat = self.precompute_tof_search_mat()  # shape (tofs, preambles, fftsize)
        self.dop_search_mat = self.precompute_dop_search_mat()  # shape (dops, preambles, fftsize)
        self.spaced_dop_vec = self.get_spaced_dop_vec()

        # internal variables
        self._estimated_noise = None
        self._v_list = None
        self._heatmaps = []

        if self.debug:
            logger.setLevel(logging.DEBUG)

        logger.debug(kwargs)

    def __str__(self):
        return json.dumps(
            {'carrier frequency (GHz)': f"{self.fc / 1e9 :.3f}",
             'bandwidth (MHz)': f"{self.bw / 1e6 :.2f}",
             'fftsize, rx_antennas, tx_antennas': f"({self.fftsize}, {self.rx_antennas}, {self.tx_antennas})",
             'antenna_spacing (meter)': f"{self.ant_spacing:.2f}",
             'aoa_search_range (radian)': f"(0, {self.aoa_max:.2f}, {self.aoa_search_step})",
             'aod_search_range (radian)': f"(0, {self.aod_max:.2f}, {self.aod_search_step})",
             'tof_search_range (nanosecond)': f"(0, {self.tof_max * 1e9:.2f}, {self.tof_search_step * 1e9:.2f})",
             'dopplershift_search_range (Hz)': f"({-self.dop_max:.2f}, {self.dop_max:.2f}, {self.dop_search_step:.2f})",
             'initial_est_stop_threshold (dB)': f"{self.initial_est_stop_threshold_db:.2f}"},
            indent=4
        )

    def precompute_aoa_search_mat(self) -> list:
        aoa_search_mat = np.array(
            [[self.get_aoa_effect(aoa_rad)] * self.preamble_repeat_cnt for aoa_rad in self.aoa_search_range])
        assert aoa_search_mat.shape == (
            len(self.aoa_search_range), self.preamble_repeat_cnt, self.fftsize, self.rx_antennas)
        return aoa_search_mat

    def precompute_aod_search_mat(self) -> list:
        aod_search_mat = np.array(
            [[self.get_aod_effect(aod_rad)] * self.preamble_repeat_cnt for aod_rad in self.aod_search_range])
        assert aod_search_mat.shape == (
            len(self.aod_search_range), self.preamble_repeat_cnt, self.fftsize, self.tx_antennas)
        return aod_search_mat

    def precompute_tof_search_mat(self) -> list:
        tof_search_mat = np.array(
            [[np.fft.ifft(self.ltf_f * self.get_tof_effect(tof))] * self.preamble_repeat_cnt for tof in
             self.tof_search_range])
        assert tof_search_mat.shape == (len(self.tof_search_range), self.preamble_repeat_cnt, self.fftsize)
        return tof_search_mat

    def precompute_dop_search_mat(self) -> list:
        """
        This saerch matrix is meant to work with preamble train.
        :return:
        """
        dop_search_mat = []
        for dop in self.dop_search_range:
            dop_mat = []
            for i in range(self.preamble_repeat_cnt):
                idx_start = i * self.preamble_interval_sample
                # aliased_dop = dop + 1 / self.preamble_interval_sec
                dop_mat.append(
                    np.exp(
                        1j * 2 * np.pi * dop / self.bw * np.arange(idx_start, idx_start + self.fftsize)) * self.ltf_t)
            dop_mat = np.array(dop_mat)
            assert dop_mat.shape == (self.preamble_repeat_cnt, self.fftsize)
            dop_search_mat.append(dop_mat)
        dop_search_mat = np.array(dop_search_mat)
        assert dop_search_mat.shape == (len(self.dop_search_range), self.preamble_repeat_cnt, self.fftsize)
        return dop_search_mat

    def get_spaced_dop_vec(self) -> np.ndarray:
        """
        No lts_t is included.
        :return:
        """
        spaced_dop_vec = []
        for dop in self.dop_search_range:
            dop_mat = []
            for i in range(self.preamble_repeat_cnt):
                idx_start = i * self.preamble_interval_sample
                # aliased_dop = dop + 1 / self.preamble_interval_sec
                dop_mat.append(np.exp(1j * 2 * np.pi * dop / self.bw * np.arange(idx_start, idx_start + self.fftsize)))
            dop_mat = np.array(dop_mat)
            assert dop_mat.shape == (self.preamble_repeat_cnt, self.fftsize)
            spaced_dop_vec.append(dop_mat)
        spaced_dop_vec = np.array(spaced_dop_vec)
        assert spaced_dop_vec.shape == (len(self.dop_search_range), self.preamble_repeat_cnt, self.fftsize)
        return spaced_dop_vec

    def gen_heatmap(self, input_sig, v_list=None, plot=False, save=''):
        assert input_sig.shape[2] == 1
        H = np.fft.fft(input_sig, axis=0) * self.ltf_f[:, np.newaxis, np.newaxis]
        res = np.zeros((len(self.aoa_search_range), len(self.tof_search_range)), dtype=np.complex)
        for ii in range(len(self.aoa_search_range)):
            for jj in range(len(self.tof_search_range)):
                # apply receive antenna array steering vector
                H_p = np.sum(H * self.aoa_search_mat[ii].conjugate()[0, :, :, np.newaxis], axis=1)
                assert H_p.shape == (self.fftsize, self.tx_antennas)
                # apply transmit antenna array steering vector
                H_pp = np.sum(H_p, axis=1)
                assert H_pp.shape == (self.fftsize,)
                # H_pp now is of shape (fftsize, rx_antennas)
                y_pp = np.fft.ifft(H_pp * self.ltf_f)
                zval = np.vdot(self.tof_search_mat[jj, 0], y_pp)  # vdot conjugates the first argument
                res[ii, jj] = zval

        if plot:
            pl.figure()
            heatmap = np.abs(res)
            pl.imshow(heatmap / heatmap.max(), aspect='auto', cmap=pl.get_cmap('jet'))
            pl.xticks(np.arange(0, len(self.tof_search_range), len(self.tof_search_range) // 10),
                      np.round(1e9 * self.tof_search_range[
                          np.arange(0, len(self.tof_search_range), len(self.tof_search_range) // 10)], 1))
            pl.yticks(np.arange(0, len(self.aoa_search_range), len(self.aoa_search_range) // 10),
                      np.round(np.rad2deg(self.aoa_search_range[
                                              np.arange(0, len(self.aoa_search_range),
                                                        len(self.aoa_search_range) // 10)]), 1))
            pl.xlabel('ToF (ns)')
            pl.ylabel('AoA (deg)')
            # ind = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            if isinstance(v_list, np.ndarray):
                for v in v_list:
                    aoa_idx = np.floor(v[0] / self.aoa_search_step)
                    tof_idx = np.floor(v[2] / self.tof_search_step)
                    pl.annotate(f"x", xy=(tof_idx, aoa_idx), fontsize='large', color='k')
                    # pl.annotate(f"x", xy=(ind[1], ind[0]), fontsize='large', color='w')
            pl.tight_layout()
            if save != '':
                pl.savefig(save)
            pl.show()
        return res

    def show_freq_response(self, sig_t, fftshift=True):
        assert sig_t.shape == (self.fftsize, self.rx_antennas, self.tx_antennas)
        ch_est = np.fft.fft(sig_t, axis=0) * self.ltf_f[:, np.newaxis, np.newaxis]
        # ch_est /= np.exp(1j*np.angle(ch_est[0,0]))
        plt, axs = pl.subplots(2, self.tx_antennas)
        axs = axs.reshape(2, self.tx_antennas)

        for tx_ant in range(self.tx_antennas):
            for rx_ant in range(self.rx_antennas):
                if fftshift:
                    ch = np.fft.fftshift(ch_est[:, rx_ant, tx_ant])
                    xticklabel = np.arange(-self.fftsize / 2, self.fftsize / 2)
                else:
                    ch = ch_est[:, rx_ant, tx_ant]
                    xticklabel = np.arange(self.fftsize)

                ch, xticklabel = ch[np.abs(ch) > 1e-12], xticklabel[np.abs(ch) > 1e-12]
                axs[0, tx_ant].plot(xticklabel, Db.mag2db(np.abs(ch)))
                axs[1, tx_ant].plot(xticklabel, np.angle(ch))

            axs[0, tx_ant].legend([f"rx_ant{i}" for i in np.arange(self.rx_antennas)])
            axs[1, tx_ant].legend([f"rx_ant{i}" for i in np.arange(self.rx_antennas)])
            axs[0, tx_ant].set_title(f"tx_ant{tx_ant}")
            axs[1, tx_ant].set_title(f"tx_ant{tx_ant}")
            axs[0, tx_ant].set_xlabel(f"subcarriers")
            axs[1, tx_ant].set_xlabel(f"subcarriers")
            axs[0, tx_ant].set_ylabel(f"Magnitude (dB)")
            axs[1, tx_ant].set_ylabel(f"Phase (radian)")
        pl.show()

    def show_heatmap(self, heatmap=None, v_list=None):
        if not isinstance(heatmap, np.ndarray):
            n = len(self._heatmaps)
            plt, axs = pl.subplots(int(np.ceil(n / 3)), 3)
            if n <= 3:
                axs = np.array([axs])
            for i in range(n):
                hmap = self._heatmaps[i]
                axs[i // 3, i % 3].imshow(np.abs(hmap), aspect='auto', cmap=pl.get_cmap('jet'))
                axs[i // 3, i % 3].set_title(f"{i + 1}-th path")
                ind = np.unravel_index(np.argmax(hmap), hmap.shape)
                axs[i // 3, i % 3].annotate(f"x ({ind[1]},{ind[0]})", xy=(ind[1], ind[0]), fontsize='large', color='w')
            pl.tight_layout()
            pl.show()
        else:
            pl.figure()
            heatmap = np.abs(heatmap)
            pl.imshow(heatmap / heatmap.max(), aspect='auto', cmap=pl.get_cmap('jet'))
            # ind = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            if v_list:
                for v in v_list:
                    pl.annotate(f"x ({ind[1]},{ind[0]})", xy=(ind[1], ind[0]), fontsize='large', color='w')
                    # pl.annotate(f"x", xy=(ind[1], ind[0]), fontsize='large', color='w')
            pl.tight_layout()
            pl.show()

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

    def get_doppler_effect(self, doppler_hz) -> np.ndarray:
        res = np.exp(1j * 2 * np.pi * doppler_hz / self.bw * np.arange(self.fftsize))
        assert res.ndim == 1 and res.shape[0] == self.fftsize
        return res

    def get_tof_effect(self, tof_s) -> np.ndarray:
        res = np.exp(-1j * 2 * np.pi * tof_s * self.sp_light / self.wavelength)
        assert res.ndim == 1 and res.shape[0] == self.fftsize
        return res

    def get_new_preamble_train_t(self, v_list, target_snr_db=None) -> np.ndarray:
        """
        Generate a single time domain channel of length ltf_t with noise added
        :param v_list: a list of path parameters
        :param target_snr_db:
        :return:
        """
        preamble_train = []
        for i in range(self.preamble_repeat_cnt):
            res = np.zeros((self.fftsize, self.rx_antennas, self.tx_antennas), dtype=np.complex)
            for v in v_list:
                aoa_rad, aod_rad, tof_s, dop_hz, alpha = v

                aoa_rad = Parameter.get_val(aoa_rad)
                aod_rad = Parameter.get_val(aod_rad)
                tof_s = Parameter.get_val(tof_s)
                dop_hz = Parameter.get_val(dop_hz)
                alpha = Parameter.get_val(alpha)

                aoa = self.get_aoa_effect(aoa_rad)  # of shape (fftsize, rx_antennas)
                aod = self.get_aod_effect(aod_rad)  # of shape (fftsize, tx_antennas)
                tof = self.get_tof_effect(tof_s)  # of shape (fftsize, )
                dop = self.get_doppler_effect(dop_hz) * np.exp(
                    1j * 2 * np.pi * dop_hz / self.bw * (i * self.preamble_interval_sample))  # of shape (fftsize, )
                channel_f = (alpha * tof * self.ltf_f)[:, np.newaxis, np.newaxis] * \
                            aoa[:, :, np.newaxis] * aod[:, np.newaxis, :]
                channel_t = np.fft.ifft(channel_f, axis=0)
                channel_t *= dop[:, np.newaxis, np.newaxis]
                res += channel_t

            if target_snr_db:
                noise = self.get_noise_mat()
                sig_pwr = Snr.get_avg_sig_pwr(res)
                res *= np.sqrt((self.per_tone_noise_mw * Db.db2mag(target_snr_db)) / sig_pwr)
                res += noise
            preamble_train.append(res)
        preamble_train = np.array(preamble_train)
        assert preamble_train.shape == (self.preamble_repeat_cnt, self.fftsize, self.rx_antennas, self.tx_antennas)
        return preamble_train

    def get_new_channel_t(self, v_list, target_snr_db=None) -> np.ndarray:
        """
        Generate a single time domain channel of length ltf_t with noise added
        :param v_list: a list of path parameters
        :param target_snr_db:
        :return:
        """
        res = np.zeros((self.fftsize, self.rx_antennas, self.tx_antennas), dtype=np.complex)
        for v in v_list:
            aoa_rad, aod_rad, tof_s, dop_hz, alpha = v

            aoa_rad = Parameter.get_val(aoa_rad)
            aod_rad = Parameter.get_val(aod_rad)
            tof_s = Parameter.get_val(tof_s)
            dop_hz = Parameter.get_val(dop_hz)
            alpha = Parameter.get_val(alpha)

            aoa = self.get_aoa_effect(aoa_rad)  # of shape (fftsize, rx_antennas)
            aod = self.get_aod_effect(aod_rad)  # of shape (fftsize, tx_antennas)
            tof = self.get_tof_effect(tof_s)  # of shape (fftsize, )
            dop = self.get_doppler_effect(dop_hz)  # of shape (fftsize, )
            channel_f = (alpha * tof * self.ltf_f)[:, np.newaxis, np.newaxis] * \
                        aoa[:, :, np.newaxis] * aod[:, np.newaxis, :]
            channel_t = np.fft.ifft(channel_f, axis=0)
            channel_t *= dop[:, np.newaxis, np.newaxis]
            res += channel_t
        assert res.shape == (self.fftsize, self.rx_antennas, self.tx_antennas)

        if target_snr_db:
            noise = self.get_noise_mat()
            sig_pwr = Snr.get_avg_sig_pwr(res)
            res *= np.sqrt((self.per_tone_noise_mw * Db.db2mag(target_snr_db)) / sig_pwr)
            res += noise

        return res

    def get_reconstructed_sig(self, v) -> np.ndarray:
        res = []

        aoa_rad, aod_rad, tof_s, dop_hz, alpha = v
        aoa_rad = Parameter.get_val(aoa_rad)
        aod_rad = Parameter.get_val(aod_rad)
        tof_s = Parameter.get_val(tof_s)
        dop_hz = Parameter.get_val(dop_hz)
        alpha = Parameter.get_val(alpha)

        aoa = self.get_aoa_effect(aoa_rad)  # of shape (fftsize, rx_antennas)
        aod = self.get_aod_effect(aod_rad)  # of shape (fftsize, tx_antennas)
        tof = self.get_tof_effect(tof_s)  # of shape (fftsize, )
        channel_f = (alpha * tof * self.ltf_f)[:, np.newaxis, np.newaxis] * \
                    aoa[:, :, np.newaxis] * aod[:, np.newaxis, :]
        channel_t = np.fft.ifft(channel_f, axis=0)

        for i in range(self.preamble_repeat_cnt):
            dop = self.get_doppler_effect(dop_hz) * np.exp(
                1j * 2 * np.pi * dop_hz / self.bw * (i * self.preamble_interval_sample))  # of shape (fftsize, )
            res.append(channel_t * dop[:, np.newaxis, np.newaxis])

        res = np.array(res)
        assert res.shape == (self.preamble_repeat_cnt, self.fftsize, self.rx_antennas, self.tx_antennas)
        return res

    def get_noise_mat(self):
        noise = np.random.randn(self.fftsize, self.rx_antennas, self.tx_antennas) + 1j * np.random.randn(
            self.fftsize, self.rx_antennas, self.tx_antennas)
        noise *= np.sqrt(self.per_tone_noise_mw) / np.abs(noise)
        return noise

    def single_path_estimation(self, y, plot=False) -> np.ndarray:
        """
        section 3.1 multi-dimensional signal estimator
        :rtype v: [AOA (rad), AOD (rad), TOF (s), DOPPLER (Hz), complex channel attenuation]
        :param y: time domain preamble received with shape (fftsize, rx_antennas, tx_antennas)
        """
        v = np.array([Parameter() for _ in range(5)])
        # frequency domain channel estimation H with shape (fftsize, rx_antennas, tx_antennas)
        H = np.fft.fft(y, axis=0) * self.ltf_f[:, np.newaxis, np.newaxis]

        maxval = 0
        heatmap = np.zeros((len(self.aoa_search_range), len(self.tof_search_range)), dtype=np.complex)
        for ii in range(len(self.aoa_search_range)):
            for jj in range(len(self.aod_search_range)):
                # apply receive antenna array steering vector
                H_p = np.sum(H * self.aoa_search_mat[ii].conjugate()[:, :, np.newaxis], axis=1)
                assert H_p.shape == (self.fftsize, self.tx_antennas)
                # apply transmit antenna array steering vector
                H_pp = np.sum(H_p * self.aod_search_mat[jj].conjugate()[:, :], axis=1)
                assert H_pp.shape == (self.fftsize,)
                # H_pp now is of shape (fftsize, rx_antennas)
                y_pp = np.fft.ifft(H_pp * self.ltf_f)
                for kk in range(len(self.dop_search_range)):
                    ds = self.dop_search_range[kk]
                    y_pp_doppler_removed = y_pp * self.get_doppler_effect(ds).conjugate()
                    for ll in range(len(self.tof_search_range)):
                        zval = np.vdot(self.tof_search_vec[ll],
                                       y_pp_doppler_removed)  # vdot conjugates the first argument
                        heatmap[ii, ll] = zval
                        if np.abs(zval) > np.abs(maxval):
                            v[0].set_val_and_idx(self.aoa_search_range[ii], ii)  # AOA
                            v[1].set_val_and_idx(self.aod_search_range[jj], jj)  # AOD
                            v[2].set_val_and_idx(self.tof_search_range[ll], ll)  # TOF
                            v[3].set_val_and_idx(self.dop_search_range[kk], kk)  # doppler shift
                            maxval = zval
        # complex channel attenuation
        complex_gain = maxval / self.rx_antennas / self.tx_antennas / np.linalg.norm(self.ltf_t) ** 2
        v[4].set_val_and_idx(complex_gain, 0)

        self._heatmaps.append(heatmap)
        if plot:
            self.show_heatmap(heatmap)

        return v

    def single_path_estimation2(self, y, plot=False) -> np.ndarray:
        """
        section 3.1 multi-dimensional signal estimator
        :rtype v: [AOA (rad), AOD (rad), TOF (s), DOPPLER (Hz), complex channel attenuation]
        :param y: time domain preamble received with shape (fftsize, rx_antennas, tx_antennas)
        """
        assert y.shape == (self.preamble_repeat_cnt, self.fftsize, self.rx_antennas, self.tx_antennas)
        v = np.array([Parameter() for _ in range(5)])
        # frequency domain channel estimation H with shape (fftsize, rx_antennas, tx_antennas)
        H = np.fft.fft(y, axis=1) * self.ltf_f[np.newaxis, :, np.newaxis, np.newaxis]

        maxval = 0
        heatmap = np.zeros((len(self.aoa_search_range), len(self.tof_search_range)), dtype=np.complex)
        # debug_aoa = []
        for ii in range(len(self.aoa_search_range)):
            # apply receive antenna array steering vector
            # aoa_search_mat.shape == (aoas, preambles, fftsize, rx_ant)
            H_p = np.sum(H * self.aoa_search_mat[ii].conjugate()[:, :, :, np.newaxis], axis=2)
            val = np.linalg.norm(H_p)
            # if self.debug:
            #     debug_aoa.append(val)
            if val > maxval:
                v[0].set_val_and_idx(self.aoa_search_range[ii], ii)  # AOA
                H_p_candidate = H_p.copy()
                maxval = val
        H_p = H_p_candidate
        assert H_p.shape == (self.preamble_repeat_cnt, self.fftsize, self.tx_antennas)

        maxval = 0
        for jj in range(len(self.aod_search_range)):
            # apply transmit antenna array steering vector
            # aod_search_mat.shape == (aods, preambles, fftsize, tx_ant)
            H_pp = np.sum(H_p * self.aod_search_mat[jj].conjugate(), axis=2)
            val = np.linalg.norm(H_pp)
            if val > maxval:
                v[1].set_val_and_idx(self.aod_search_range[jj], jj)  # AOD
                H_pp_candidate = H_pp.copy()
                maxval = val
        H_pp = H_pp_candidate
        assert H_pp.shape == (self.preamble_repeat_cnt, self.fftsize)

        y_pp = np.fft.ifft(H_pp * self.ltf_f[np.newaxis, :], axis=1)
        assert y_pp.shape == (self.preamble_repeat_cnt, self.fftsize)

        maxval = 0
        for ll in range(len(self.tof_search_range)):
            zval = np.vdot(self.tof_search_mat[ll], y_pp)  # vdot conjugates the first argument
            if np.abs(zval) > np.abs(maxval):
                v[2].set_val_and_idx(self.tof_search_range[ll], ll)  # TOF
                maxval = zval

        # y_pp_tof_removed = y_pp * self.tof_search_mat[v[2].idx].conjugate()
        # assert y_pp_tof_removed.shape == (self.preamble_repeat_cnt, self.fftsize)

        maxval = 0
        plot_dop = []
        for kk in range(len(self.dop_search_range)):
            zval = np.vdot(self.dop_search_mat[kk], y_pp)  # vdot conjugates the first argument
            plot_dop.append(abs(zval))
            if np.abs(zval) > np.abs(maxval):
                v[3].set_val_and_idx(self.dop_search_range[kk], kk)  # doppler shift
                maxval = zval

        # pl.figure()
        # pl.plot(self.dop_search_range, plot_dop)
        # pl.show()

        # complex channel attenuation
        maxval = np.vdot(self.tof_search_mat[v[2].idx] * self.spaced_dop_vec[v[3].idx], y_pp)
        complex_gain = maxval / self.preamble_repeat_cnt / self.rx_antennas / self.tx_antennas / np.linalg.norm(
            self.ltf_t) ** 2
        v[4].set_val_and_idx(complex_gain, 0)

        self._heatmaps.append(heatmap)
        if plot:
            self.show_heatmap(heatmap)

        return v

    def coordinate_descent(self, y, v_this_path) -> np.ndarray:
        """
        section 3.2.2 iterative path parameter refinement
        :param y: time domain preamble received with size fftsize x rx_antennas
        """
        assert y.shape == (self.preamble_repeat_cnt, self.fftsize, self.rx_antennas, self.tx_antennas)
        v = np.array([Parameter() for _ in range(5)])

        # frequency domain channel estimation H with shape (fftsize, rx_antennas, tx_antennas)
        H = np.fft.fft(y, axis=1) * self.ltf_f[np.newaxis, :, np.newaxis, np.newaxis]

        # 1. search along AOA dimension
        aoa_rad_orig, aod_rad_orig, tof_s_orig, dop_hz_orig, alpha_orig = v_this_path
        maxval = 0
        for ii in range(len(self.aoa_search_range)):
            H_p = np.sum(H * self.aoa_search_mat[ii].conjugate()[:, :, :, np.newaxis], axis=2)
            H_pp = np.sum(H_p * self.aod_search_mat[aod_rad_orig.idx].conjugate(), axis=2)
            y_pp = np.fft.ifft(H_pp * self.ltf_f[np.newaxis, :], axis=1)
            zval = np.vdot(self.tof_search_mat[tof_s_orig.idx] * self.spaced_dop_vec[dop_hz_orig.idx], y_pp)
            if np.abs(zval) > np.abs(maxval):
                v[0].set_val_and_idx(self.aoa_search_range[ii], ii)  # AOA
                maxval = zval

        # 2. search along AOD dimension
        aoa_rad_new = v[0]
        maxval = 0
        for ii in range(len(self.aod_search_range)):
            H_p = np.sum(H * self.aoa_search_mat[aoa_rad_new.idx].conjugate()[:, :, :, np.newaxis], axis=2)
            H_pp = np.sum(H_p * self.aod_search_mat[ii].conjugate(), axis=2)
            y_pp = np.fft.ifft(H_pp * self.ltf_f[np.newaxis, :], axis=1)
            zval = np.vdot(self.tof_search_mat[tof_s_orig.idx] * self.spaced_dop_vec[dop_hz_orig.idx], y_pp)
            if np.abs(zval) > np.abs(maxval):
                v[1].set_val_and_idx(self.aod_search_range[ii], ii)  # AOD
                maxval = zval

        # 3. search along TOF dimension
        aod_rad_new = v[1]
        maxval = 0
        for ii in range(len(self.tof_search_range)):
            H_p = np.sum(H * self.aoa_search_mat[aoa_rad_new.idx].conjugate()[:, :, :, np.newaxis], axis=2)
            H_pp = np.sum(H_p * self.aod_search_mat[aod_rad_new.idx].conjugate(), axis=2)
            y_pp = np.fft.ifft(H_pp * self.ltf_f[np.newaxis, :], axis=1)
            zval = np.vdot(self.tof_search_mat[ii] * self.spaced_dop_vec[dop_hz_orig.idx], y_pp)
            if np.abs(zval) > np.abs(maxval):
                v[2].set_val_and_idx(self.tof_search_range[ii], ii)  # TOF
                maxval = zval

        # 4. search along Doppler dimension
        tof_s_new = v[2]
        maxval = 0
        for ii in range(len(self.dop_search_range)):
            H_p = np.sum(H * self.aoa_search_mat[aoa_rad_new.idx].conjugate()[:, :, :, np.newaxis], axis=2)
            H_pp = np.sum(H_p * self.aod_search_mat[aod_rad_new.idx].conjugate(), axis=2)
            y_pp = np.fft.ifft(H_pp * self.ltf_f[np.newaxis, :], axis=1)
            zval = np.vdot(self.tof_search_mat[tof_s_new.idx] * self.spaced_dop_vec[ii], y_pp)
            if np.abs(zval) > np.abs(maxval):
                v[3].set_val_and_idx(self.dop_search_range[ii], ii)  # Doppler
                maxval = zval

                # 5. compute complex channel attenuation
        complex_gain = maxval / self.preamble_repeat_cnt / self.rx_antennas / self.tx_antennas / np.linalg.norm(
            self.ltf_t) ** 2
        v[4].set_val_and_idx(complex_gain, 0)

        return v

    def resolve_multipath(self, input_sig) -> np.ndarray:
        """
        Takes single ltf_t as input.
        section 3.2 in the paper
        :param y: time domain preamble received with size fftsize x rx_antennas
        """
        assert input_sig.shape == (self.preamble_repeat_cnt, self.fftsize, self.rx_antennas, self.tx_antennas)
        # v_list, recon_sig_t = self.initial_estimation(input_sig)
        v_list, recon_sig_t = self.initial_estimation2(input_sig)

        # 2. iterative estimation refinement
        # iteratively refine the v_list parameters
        it = 0
        while True:
            v_list_new = []
            recon_sig_t_new = []
            for i in range(len(v_list)):
                v_new = self.coordinate_descent(recon_sig_t[i] + self._estimated_noise, v_list[i])
                v_list_new.append(v_new)
                recon_sig_t_new.append(self.get_reconstructed_sig(v_new))
                self._estimated_noise += recon_sig_t[i] - recon_sig_t_new[i]

            v_list_new = np.array(v_list_new)
            it += 1
            logger.debug(f"refinement iteration {it:d} {self.print_pretty(v_list_new)}")
            if np.any(v_list_new - v_list != 0):
                v_list = v_list_new
                recon_sig_t = recon_sig_t_new
            else:
                break

        self._v_list = v_list
        return v_list

    def resolve_multipath_preamble_train(self, preamble_train) -> np.ndarray:
        """
        Takes single ltf_t as input.
        section 3.2 in the paper
        :param y: time domain preamble received with size fftsize x rx_antennas
        """
        v_list_train = []
        for i in range(21):
            v_list = self.resolve_multipath(preamble_train[i:i + 20, :, :, :])
            print(self.print_pretty(v_list))
            v_list_train.append(v_list)
        v_list_train = np.array(v_list_train)
        v_list_train_new = v_list_train.copy()

        # # use preamble train for doppler estimation
        # for i, v_list in enumerate(v_list_train):
        #     for j, v in enumerate(v_list):
        #         y_pp_train = []
        #         for input_sig in preamble_train:
        #             H = np.fft.fft(input_sig, axis=0) * self.ltf_f[:, np.newaxis, np.newaxis]
        #             aoa, aod, tof, dop, mag = v
        #             # apply receive antenna array steering vector
        #             H_p = np.sum(H * self.aoa_search_mat[aoa.idx].conjugate()[:, :, np.newaxis], axis=1)
        #             assert H_p.shape == (self.fftsize, self.tx_antennas)
        #             # apply transmit antenna array steering vector
        #             H_pp = np.sum(H_p * self.aod_search_mat[aod.idx].conjugate()[:, :], axis=1)
        #             assert H_pp.shape == (self.fftsize,)
        #             # H_pp now is of shape (fftsize, rx_antennas)
        #             y_pp = np.fft.ifft(H_pp * self.ltf_f)
        #             y_pp_train.append(y_pp)
        #         y_pp_train = np.array(y_pp_train)
        #         cor = [np.abs(np.vdot(x, y_pp_train)) for x in self.dop_search_mat]
        #         dop_idx = np.argmax(cor)
        #         v_list_train_new[i][j][3].set_val_and_idx(self.dop_search_range[dop_idx], dop_idx)  # doppler shift
        #         pl.figure()
        #         pl.plot(cor)
        #         pl.show()
        #     exit()

        return v_list_train_new

    def initial_estimation(self, input_sig):
        """
        section 3.2.1 initial estimation. Takes single ltf_t as input.
        :param input_sig:
        :return:
        """
        input_sig = input_sig.copy()
        self._heatmaps = []
        v_list = []
        recon_sig_t = []
        logger.debug(f"Initial SNR (dB): {Snr.get_avg_snr_db(input_sig, self.per_tone_noise_mw):.2f} "
                     f"Min SNR (dB): {Snr.get_min_snr_db(input_sig, self.per_tone_noise_mw):.2f}")

        # 1. initialization, successive interference cancellation
        while True:
            v = self.single_path_estimation(input_sig)
            v_list.append(v)
            # aoa, aod, tof, dop, alpha = v
            recon_sig_t.append(self.get_reconstructed_sig(v))
            input_sig -= recon_sig_t[-1]
            residual_snr_db = Snr.get_avg_snr_db(input_sig, self.per_tone_noise_mw)
            min_snr_db = Snr.get_min_snr_db(input_sig, self.per_tone_noise_mw)
            logger.debug(
                f"Canceling path {len(recon_sig_t):>2d}, residual SNR: {residual_snr_db:.2f}, min SNR {min_snr_db:.2f}" \
                f", tof(ns) {Parameter.get_val(v[2]) * 1e9:.3f}, aoa(deg) {np.rad2deg(Parameter.get_val(v[0])):.3f}")

            if residual_snr_db < self.initial_est_stop_threshold_db:
                self._estimated_noise = input_sig
                break
        v_list = np.array(v_list)
        logger.debug(f"Initialization {self.print_pretty(v_list)}")
        return v_list, recon_sig_t

    def initial_estimation2(self, input_sig):
        """
        section 3.2.1 initial estimation. Takes single ltf_t as input.
        :param input_sig:
        :return:
        """
        assert input_sig.shape == (self.preamble_repeat_cnt, self.fftsize, self.rx_antennas, self.tx_antennas)
        input_sig = input_sig.copy()
        self._heatmaps = []
        v_list = []
        recon_sig_t = []
        # logger.debug(f"Initial SNR (dB): {Snr.get_avg_snr_db(input_sig, self.per_tone_noise_mw):.2f} "
        #              f"Min SNR (dB): {Snr.get_min_snr_db(input_sig, self.per_tone_noise_mw):.2f}")
        last_residual_snr_db = Snr.avg_rf_chain_snr_db(input_sig, self.rf_chain_noise_pwr)
        last_min_residual_snr_db = Snr.min_rf_chain_snr_db(input_sig, self.rf_chain_noise_pwr)
        assert last_residual_snr_db.shape == (self.rx_antennas,) and last_min_residual_snr_db.shape == (
            self.rx_antennas,)
        logger.debug(f"Initial SNR (dB): {np.round(last_residual_snr_db, 2)}, "
                     f"Min SNR (dB): {np.round(last_min_residual_snr_db, 2)}")

        # 1. initialization, successive interference cancellation
        while True:
            v = self.single_path_estimation2(input_sig)

            # aoa, aod, tof, dop, alpha = v
            sig_t = self.get_reconstructed_sig(v)
            input_sig -= sig_t
            # residual_snr_db = Snr.get_avg_snr_db(input_sig, self.per_tone_noise_mw)
            # min_snr_db = Snr.get_min_snr_db(input_sig, self.per_tone_noise_mw)
            residual_snr_db = Snr.avg_rf_chain_snr_db(input_sig, self.rf_chain_noise_pwr)
            min_snr_db = Snr.min_rf_chain_snr_db(input_sig, self.rf_chain_noise_pwr)
            snr_reduction = last_residual_snr_db - residual_snr_db
            last_residual_snr_db, last_min_residual_snr_db = residual_snr_db, min_snr_db
            logger.debug(f"Canceling path {len(recon_sig_t):>2d}, "
                         f"residual SNR: {np.round(residual_snr_db, 2)}, "
                         f"min SNR {np.round(min_snr_db, 2)}, "
                         f"SNR reduction {np.round(snr_reduction, 3)}")
            # print(self.print_pretty(np.array(v_list)))

            if (len(v_list) > 0 and np.all(snr_reduction < 2)):# or len(v_list) > 1:
                self._estimated_noise = input_sig
                break

            v_list.append(v)
            recon_sig_t.append(sig_t)

            if np.all(residual_snr_db < self.initial_est_stop_threshold_db):
                self._estimated_noise = input_sig
                break
        v_list = np.array(v_list)
        logger.debug(f"Initialization {self.print_pretty(v_list)}")
        return v_list, recon_sig_t


def coordinate_descent(y, LTS, ant_sp, FFT_SZ, K, l1, aoa_range, tof_range, aoa_matrix, aod_matrix, U, old_tof_idx,
                       plot=False, debug=False):
    Y = np.fft.fft(y, axis=0) * LTS.reshape(-1, 1)
    maxval = 0
    for ii in range(len(aoa_range)):
        H_p = Y * aoa_matrix[ii]
        H_pp = np.sum(H_p * aod_matrix, axis=1)
        y_pp = np.fft.ifft(H_pp * LTS)
        zval = np.abs(np.vdot(U[old_tof_idx], y_pp))  # vdot conjugates the first argument
        if zval > maxval:
            maxval = zval
            new_aoa_idx = ii

    maxval = 0
    H_p = Y * aoa_matrix[new_aoa_idx]
    H_pp = np.sum(H_p * aod_matrix, axis=1)
    y_pp = np.fft.ifft(H_pp * LTS)
    for ii in range(len(tof_range)):
        zval = np.vdot(U[ii], y_pp)  # vdot conjugates the first argument
        if np.abs(zval) > np.abs(maxval):
            maxval = zval
            new_tof_idx = ii

    ALPHA = maxval / K / np.linalg.norm(np.fft.ifft(LTS)) ** 2

    if plot:
        pl.figure()
        pl.imshow(np.abs(heatmap), aspect='auto', cmap=pl.get_cmap('jet'))
        pl.show()
    return np.array([aoa_range[new_aoa_idx], np.pi / 2, tof_range[new_tof_idx], 0.0, ALPHA])


def md_estimator(y, LTS, ant_sp, FFT_SZ, K, l1, aoa_range, tof_range, aoa_matrix, aod_matrix, U, plot=False,
                 debug=False):
    Y = np.fft.fft(y, axis=0) * LTS.reshape(-1, 1)

    maxval = 0
    heatmap = np.zeros((len(aoa_range), len(tof_range)), dtype=np.complex)
    '''
    ii=0
    for aoa_rad in aoa_range:
        jj=0
        for tof in tof_range:
            aoa_matrix = np.exp(1j*2*np.pi*ant_sp*np.arange(K)*np.cos(aoa_rad)/l1.reshape(-1,1))
            H_p = Y*aoa_matrix
            aod_matrix = np.exp(1j*2*np.pi*ant_sp*np.arange(K)*np.cos(np.pi/2)/l1.reshape(-1,1))
            H_pp = np.sum(H_p*aod_matrix, axis=1)
            y_pp = ifft(H_pp*LTS)
            U = ifft(LTS*np.exp(-1j*2*np.pi*tof*3e8/l1))
            zval = np.dot(y_pp, U.conj())
            heatmap[ii,jj] = zval
            if np.abs(zval) > np.abs(maxval):
                TOF = tof
                AOA_RAD = aoa_rad
                maxval = zval
            jj+=1
        ii+=1
    '''

    for ii in range(len(aoa_range)):
        H_p = Y * aoa_matrix[ii]
        H_pp = np.sum(H_p * aod_matrix, axis=1)
        y_pp = np.fft.ifft(H_pp * LTS)
        for jj in range(len(tof_range)):
            zval = np.vdot(U[jj], y_pp)  # vdot conjugates the first argument
            heatmap[ii, jj] = zval
            if np.abs(zval) > np.abs(maxval):
                TOF = tof_range[jj]
                AOA_RAD = aoa_range[ii]
                maxval = zval

    ALPHA = maxval / K / np.linalg.norm(np.fft.ifft(LTS)) ** 2

    if plot:
        pl.figure()
        pl.imshow(np.abs(heatmap), aspect='auto', cmap=pl.get_cmap('jet'))
        pl.show()
    return np.array([AOA_RAD, np.pi / 2, TOF, 0.0, ALPHA])


def decompose(input_sig, noise_pwr, rescale, LTS, ant_sp, FFT_SZ, K, l1, aoa_step, tof_step, aoa_range, tof_range,
              aoa_matrix, aod_matrix, U, plot=False, debug=False):
    # v: estimated signal component vector
    # v = [AOA (rad), AOD (rad), TOF (s), DOPPLER, ALPHA (channel attenuation)]
    v = []
    recon_sig = []
    if debug:
        sig_pwr = np.sum(np.abs(input_sig) ** 2) / FFT_SZ / K
        print('Initial SNR:', 10 * np.log10(sig_pwr / noise_pwr))
    # SIC initialization
    while True:
        v.append(md_estimator(input_sig, LTS, ant_sp, FFT_SZ, K, l1, aoa_range, tof_range, aoa_matrix, aod_matrix, U,
                              plot=False, debug=False))
        s = v[-1][4] * np.exp(-1j * 2 * np.pi * v[-1][2] * 3e8 / l1).reshape(-1, 1) * np.exp(
            -1j * 2 * np.pi * ant_sp * np.arange(K) * np.cos(v[-1][0]) / l1.reshape(-1, 1)) * LTS.reshape(-1, 1)
        recon_sig.append(np.fft.ifft(s, axis=0))
        input_sig -= recon_sig[-1]
        sig_pwr = np.sum(np.abs(input_sig) ** 2) / FFT_SZ / K
        min_sig_pwr = np.min(np.sum(np.abs(input_sig) ** 2, axis=1) / K)
        if debug:
            print('cancel', len(recon_sig), 'paths,', 'residual SNR:', Db.mag2db(sig_pwr / noise_pwr), 'min SNR:',
                  Db.mag2db(min_sig_pwr / noise_pwr))
        if 10 * np.log10(sig_pwr / noise_pwr) < 1:
            W = input_sig
            break
    v = np.array(v)

    if debug:
        print('======================')
        print('Initialization result:')
        print('AoA', v[:, 0] / np.pi * 180)
        print('Dist', v[:, 2] * 3e8)
        print('alpha (mag)', np.abs(v[:, 4]))
        print('alpha (rad)', np.angle(v[:, 4]))

    # iteration step
    it = 1
    while True:
        v_p = []
        recon_sig_p = []
        for ii in range(len(recon_sig)):
            # coordinate descent
            old_tof_idx = int(np.real(v[ii][2] / tof_step))
            v_p.append(
                coordinate_descent(recon_sig[ii] + W, LTS, ant_sp, FFT_SZ, K, l1, aoa_range, tof_range, aoa_matrix,
                                   aod_matrix, U, old_tof_idx, plot=False, debug=False))
            s = v_p[-1][4] * np.exp(-1j * 2 * np.pi * v_p[-1][2] * 3e8 / l1).reshape(-1, 1) * np.exp(
                -1j * 2 * np.pi * ant_sp * np.arange(K) * np.cos(v_p[-1][0]) / l1.reshape(-1, 1)) * LTS.reshape(-1, 1)
            recon_sig_p.append(np.fft.ifft(s, axis=0))
            W += recon_sig[ii] - recon_sig_p[ii]

        v_p = np.array(v_p)
        if debug:
            print('======================')
            print('iteration', it)
            table = {'AoA': np.real(v[:, 0] / np.pi * 180), 'Dist': np.real(v[:, 2] * 3e8), 'Alpha:': np.abs(v[:, 4]),
                     'Alpha(rad):': np.angle(v[:, 4])}
            print(tabulate(table, headers='keys', tablefmt='github'))
            table = {'AoA': np.real(v_p[:, 0] / np.pi * 180), 'Dist': np.real(v_p[:, 2] * 3e8),
                     'Alpha:': np.abs(v_p[:, 4]), 'Alpha(rad):': np.angle(v_p[:, 4])}
            print('\n' + tabulate(table, headers='keys', tablefmt='github'))

        it += 1
        if np.any(np.abs(v[:, 0] - v_p[:, 0]) >= aoa_step) or np.any(np.abs(v[:, 2] - v_p[:, 2]) >= tof_step):
            v = v_p
            recon_sig = recon_sig_p
        else:
            break

            # remove rescaling
    v[:, 4] /= rescale
    return v


def old_main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s :: %(levelname)s :: %(message)s')
    # kwargs = {'rx_antennas': 4, 'tx_antennas': 1, 'aod_max': 1e-6, 'dopplershift_max': 0.1,
    #           'debug': True}
    # a = Mdtrack(**kwargs)
    # print(a)
    # exit()
    # ch = a.get_channel_t([ [60 / 180 * np.pi, 0.0, 48.3e-9, 0.0, 0.6],
    #                        [153.8 / 180 * np.pi, 0.0, 87.3e-9, 0.0, 0.3]])
    # print(ch.shape)
    # est = a.initial_estimation(ch, plot=False)
    # # print(est)
    # a.resolve_multipath(ch)
    # exit()

    LTS = np.array(
        [(1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j), (1 + 0j), (1 + 0j), (1 + 0j),
         (1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j),
         (-1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j), (1 + 0j), (1 + 0j), (1 + 0j), (1 + 0j),
         (1 + 0j), (1 + 0j), 0j, (1 + 0j), (-1 + 0j), (1 + 0j), (1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j), (1 + 0j),
         (1 + 0j), (-1 + 0j), (1 + 0j), (1 + 0j), (1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j),
         (-1 + 0j), (1 + 0j), (-1 + 0j), (1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j), (-1 + 0j),
         (-1 + 0j), (-1 + 0j), (1 + 0j)])
    # LTS = np.ones(64)
    LTS = np.fft.ifftshift(LTS)
    LTS_T = np.fft.ifft(LTS)
    # CFO = 15687
    # preamble = np.concatenate((LTS_T, LTS_T))
    # preamble *= np.exp(1j*2*np.pi*CFO/20e6*np.arange(len(preamble)))
    # cfo_est = [preamble[i].conjugate()*preamble[i+len(LTS_T)] for i in range(len(LTS_T))]
    # print(np.angle(cfo_est))
    # cfo_est = np.mean(cfo_est)
    # print(np.abs(cfo_est), np.angle(cfo_est))
    # cfo_est = np.angle(cfo_est)
    # print('estimated CFO:', cfo_est/2/np.pi/len(LTS_T)*20e6,'True CFO:', CFO)
    # exit()
    bw = 20e6
    SNR = 20
    FFT_SZ = len(LTS)
    aoa_step = 0.02  # rad
    tof_step = 0.5e-9
    tof_max = 150e-9
    aoa_range = np.arange(0, np.pi, aoa_step)
    tof_range = np.arange(0, tof_max, tof_step)
    K = 4
    fc = 2412e6
    l1 = 3e8 / (np.fft.fftfreq(FFT_SZ) * bw + fc)
    ant_sp = 3e8 / fc / 2
    d1 = 6.18  # 20.6 ns
    d2 = 8.43  # 28.1 ns
    theta1 = 60.7  # deg
    theta2 = 73.4
    a1 = 0.7
    a2 = 0.22

    # precompute aoa matrix
    aoa_matrix = [np.exp(1j * 2 * np.pi * ant_sp * np.arange(K) * np.cos(aoa_rad) / l1.reshape(-1, 1)) for aoa_rad in
                  aoa_range]
    # precompute aod matrix
    aod_matrix = np.exp(1j * 2 * np.pi * ant_sp * np.arange(K) * np.cos(np.pi / 2) / l1.reshape(-1, 1))
    # precompute delayed LTS_T for correlation
    U = [np.fft.ifft(LTS * np.exp(-1j * 2 * np.pi * tof * 3e8 / l1)) for tof in tof_range]

    AR = a1 * np.exp(-1j * 2 * np.pi * d1 / l1).reshape(-1, 1) * np.exp(
        -1j * 2 * np.pi * ant_sp * np.arange(K) * np.cos(theta1 / 180 * np.pi) / l1.reshape(-1, 1))
    AR += a2 * np.exp(-1j * 2 * np.pi * d2 / l1).reshape(-1, 1) * np.exp(
        -1j * 2 * np.pi * ant_sp * np.arange(K) * np.cos(theta2 / 180 * np.pi) / l1.reshape(-1, 1))
    AR = AR * LTS.reshape(-1, 1)
    y = np.fft.ifft(AR, axis=0)

    ###
    sig_pwr = np.sum(np.abs(y) ** 2) / FFT_SZ / K
    noise_pwr = 1e-5  # sig_pwr*10**(-SNR/10)
    noise = np.random.randn(FFT_SZ, K) + 1j * np.random.randn(FFT_SZ, K)
    noise = np.sqrt(noise_pwr) * noise / np.abs(noise)
    rescale = np.sqrt((noise_pwr * 10 ** (SNR / 10)) / sig_pwr)
    y *= rescale
    y += noise
    input_sig = y

    t1 = time.time()
    v = decompose(input_sig, noise_pwr, rescale, LTS, ant_sp, FFT_SZ, K, l1, aoa_step, tof_step, aoa_range, tof_range,
                  aoa_matrix, aod_matrix, U, plot=False, debug=True)
    t2 = time.time()

    table = {'AoA': np.real(v[:, 0] / np.pi * 180), 'Dist': np.real(v[:, 2] * 3e8), 'Alpha:': np.abs(v[:, 4]),
             'Alpha(rad):': np.angle(v[:, 4])}
    logging.info('Iteration result:\n' + tabulate(table, headers='keys', tablefmt='github'))
    logging.info('Run Time: {0:.3f}'.format(t2 - t1))

    # print('AoA', v[:,0]/np.pi*180)
    # print('Dist', v[:,2]*3e8)
    # print('alpha', np.abs(v[:,4]))
    # print(v[0][0]/np.pi*180)
    # print('SNR:', 10*np.log10(sig_pwr/noise_pwr))
    # print('residual SNR:', 10*np.log10(sig_pwr/noise_pwr))


if __name__ == "__main__":
    pass
