import numpy as np
from ieee80211.preamble import Preamble
from utils.common import Db, Snr

class Signal(object):
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
        self.per_tone_noise_mw = kwargs.get('per_tone_noise_mw', 1e-5)
        self.fftsize = len(self.ltf_f)
        self.wavelength = self.sp_light / (np.fft.fftfreq(self.fftsize) * self.bw + self.fc)

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

    def get_noise_mat(self):
        noise = np.random.randn(self.fftsize, self.rx_antennas, self.tx_antennas) + 1j * np.random.randn(
            self.fftsize, self.rx_antennas, self.tx_antennas)
        noise *= np.sqrt(self.per_tone_noise_mw) / np.abs(noise)
        return noise

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

            # aoa_rad = Parameter.get_val(aoa_rad)
            # aod_rad = Parameter.get_val(aod_rad)
            # tof_s = Parameter.get_val(tof_s)
            # dop_hz = Parameter.get_val(dop_hz)
            # alpha = Parameter.get_val(alpha)

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

    def get_new_channel_f(self, v_list, target_snr_db=None) -> np.ndarray:
        """
        Generate a single freq domain channel of length ltf_f with noise added
        :param v_list: a list of path parameters
        :param target_snr_db:
        :return:
        """
        sig_t = self.get_new_channel_t(v_list, target_snr_db)
        sig_f = np.fft.fft(sig_t, axis=0) * self.ltf_f[:, np.newaxis, np.newaxis]
        return sig_f