import functools
import time
import logging
import subprocess
import numpy as np

logger = logging.getLogger(__name__)

class Snr:
    @staticmethod
    def get_avg_sig_pwr(sig_t):
        """
        Average power across antennas and subcarriers
        :param sig_t: ndarray of shape (fftsize, rx_antennas, tx_antennas)
        :return:
        """
        return np.mean(np.abs(sig_t) ** 2)

    @staticmethod
    def get_avg_snr_db(sig_t, per_tone_noise_mw):
        """
        Compute SNR from time domain samples
        :param sig_t: ndarray of shape (fftsize, rx_antennas, tx_antennas)
        :param per_tone_noise_mw:
        :return:
        """
        sig_t_reshaped = sig_t.reshape(sig_t.shape[0], -1)
        avg_per_tone_sig_pwr = np.mean(np.abs(sig_t_reshaped) ** 2, axis=1)
        per_tone_snr = avg_per_tone_sig_pwr / per_tone_noise_mw
        # logger.info(f"per_tone_snr_db \n{', '.join([f'{Db.mag2db(x):>4.1f}' for x in per_tone_snr])}")
        return Db.mag2db(np.mean(per_tone_snr))

    @staticmethod
    def get_min_snr_db(sig_t, per_tone_noise_mw):
        """
        Compute SNR from time domain samples
        :param sig_t: ndarray of shape (fftsize, rx_antennas, tx_antennas)
        :param per_tone_noise_mw:
        :return:
        """
        sig_t_reshaped = sig_t.reshape(sig_t.shape[0], -1)
        avg_per_tone_sig_pwr = np.mean(np.abs(sig_t_reshaped) ** 2, axis=1)
        per_tone_snr = avg_per_tone_sig_pwr / per_tone_noise_mw
        # logger.info(f"per_tone_snr_db \n{', '.join([f'{Db.mag2db(x):>4.1f}' for x in per_tone_snr])}")
        return Db.mag2db(np.min(per_tone_snr))


class Db:
    @staticmethod
    def db2mag(x):
        return 10 ** (x / 10.0)

    @staticmethod
    def mag2db(x):
        return 10 * np.log10(x)


def execute_cmd(cmd, timeout=10):
    p = subprocess.run(cmd, shell=True, check=True, timeout=timeout)
    return p.stdout


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)
        t2 = time.time()
        logger.info(f"Executed {func.__name__} in {t2 - t1:.4f} second.")
        return res, t2 - t1

    return wrapper