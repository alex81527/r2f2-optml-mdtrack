import numpy as np
import multiprocessing as mp
from tensorflow import keras
import scipy.ndimage
import scipy.sparse
from utils.siggen import Signal
import bisect


def data_normalization(sig_f):
    res = sig_f[1:, :, :] / np.exp(1j * np.angle(sig_f[1, 0, 0]))
    res = res/np.abs(res).max()
    return res


def data_vectorization(sig_f):
    x = sig_f.ravel()
    return np.concatenate((np.real(x), np.imag(x)))


def map_multiprocess(cores, ufunc, *args):
    pool = mp.Pool(processes=cores)
    if isinstance(args[0], int):
        tasks_per_proc = np.ceil(args[0] / cores).astype(int)
        asyncresults = [pool.apply_async(ufunc, args=(tasks_per_proc, *args[1:])) for i in range(cores)]
    else:
        tasks_per_proc = np.ceil(len(args[0]) / cores).astype(int)
        asyncresults = [pool.apply_async(ufunc, args=(args[0][i * tasks_per_proc:(i + 1) * tasks_per_proc], *args[1:]))
                        for i in range(cores)]
    pool.close()
    pool.join()
    res = []
    for r in asyncresults:
        res.extend(r.get())
    pool.terminate()
    return np.array(res)

def gen_1d_tof_physical_paths(size, resolution, bw, tof_max, max_physical_paths):
    res = []
    # generate tofs that are at least (sampling_interval * resolution) apart
    choices_for_tof = np.arange(0, tof_max, resolution / bw)
    for i in range(size):
        npath = np.random.randint(1, max_physical_paths + 1)
        tofs = np.random.choice(choices_for_tof, npath, replace=False)
        adjustable = min(tofs.min(), tof_max - tofs.max())
        sign = 1 if np.random.rand() > 0.5 else -1
        random_adjust = sign * np.random.rand() * adjustable
        tofs += random_adjust

        v_list = np.zeros((npath, 5), dtype=np.complex)
        # aoa, aod, tof, dop, complex gain
        v_list[:, 2] = tofs
        v_list[:, 4] = np.random.rand(npath) * np.exp(1j * 2 * np.pi * np.random.rand(npath))
        res.append(v_list)

    return res

def gen_2d_aoa_tof_physical_paths(size, resolution, bw, tof_max, rx_antennas, max_physical_paths):
    res = []
    # generate tofs that are at least (resolution/bw) apart
    choices_for_tof = np.arange(0, tof_max, 2 * resolution / bw)
    choices_for_tof2 = np.arange(resolution / bw, tof_max, 2 * resolution / bw)
    choices_for_aoa_cos = np.arange(-1, 1, resolution * 2 / rx_antennas)
    for i in range(size):
        npath = np.random.randint(2, max_physical_paths + 1)
        assert len(choices_for_tof) >= npath, f"choices for tof {choices_for_tof}, npath={npath}"
        if np.random.rand() > 0.5:
            tofs = np.random.choice(choices_for_tof, npath, replace=False)
        else:
            tofs = np.random.choice(choices_for_tof2, npath, replace=False)
        # adjustable = min(tofs.min(), tof_max - tofs.max())
        # sign = 1 if np.random.rand() > 0.5 else -1
        # random_adjust = sign * np.random.rand() * adjustable
        random_adjust = np.random.rand(npath) * resolution / bw
        tofs += random_adjust
        # tofs = np.random.rand(npath) * tof_max

        # aoa_coss = np.random.choice(choices_for_aoa_cos, npath, replace=False)
        # adjustable = min(aoa_coss.min() + 1, 1 - aoa_coss.max())
        # sign = 1 if np.random.rand() > 0.5 else -1
        # random_adjust = sign * np.random.rand() * adjustable
        # aoa_coss += random_adjust
        aoa_rads = np.random.rand(npath) * np.pi

        # the last path is intended for mobile path
        v_list = np.zeros((npath, 5), dtype=np.complex)
        # aoa, aod, tof, dop, complex gain
        v_list[:, 0] = aoa_rads
        v_list[:, 2] = tofs
        # v_list[:-1, 4] = np.random.uniform(low=1/10, high=1, size=npath-1) * np.exp(1j * 2 * np.pi * np.random.rand(npath-1))
        # v_list[-1, 4] = np.random.uniform(low=1/20, high=1/10, size=1) * np.exp(1j * 2 * np.pi * np.random.rand())
        v_list[:, 4] = np.random.rand(npath) * np.exp(1j * 2 * np.pi * np.random.rand(npath))
        res.append(v_list)

    return res


def from_path_to_channel(v_lists, snr_range, sig_generator):
    res = []
    for v_list in v_lists:
        snr = snr_range[0] + np.random.rand() * (snr_range[1] - snr_range[0])
        sig_f = sig_generator.get_new_channel_f(v_list, target_snr_db=snr)
        # leave out dc carrier and rescale st. sig_f[0,0,0] has zero phase
        sig_f_normalized = data_normalization(sig_f)
        sig_f_vectorized = data_vectorization(sig_f_normalized)
        res.append(sig_f_vectorized)
    return res

def from_path_to_1d_tof_dense_target(v_lists, tof_search_range, tof_search_step, resolution, bw):
    res = []
    for v_list in v_lists:
        target_y = np.zeros(len(tof_search_range))
        attenuations = np.abs(v_list[:, 4])
        tof_idxs = [bisect.bisect_left(tof_search_range, i) if i < tof_search_range[-1] else len(
            tof_search_range) - 1 for i in v_list[:, 2]]
        target_y[tof_idxs] = attenuations
        sigma_tof = resolution / bw / tof_search_step / 3
        target_y = scipy.ndimage.gaussian_filter1d(target_y, sigma_tof) * np.sqrt(2 * np.pi) * sigma_tof
        res.append(target_y)
    return res


def from_path_to_2d_aoa_tof_sparse_target(v_lists, tof_search_range, tof_search_step, aoa_cos_search_range,
                                          aoa_cos_search_step):
    res = []
    for v_list in v_lists:
        # target_y = np.zeros((len(tof_search_range), len(aoa_cos_search_range)))
        attenuations = np.abs(v_list[:, 4])
        attenuations = np.where(attenuations<0.1, 0.1, attenuations)
        tof_idxs = [bisect.bisect_left(tof_search_range, i) if i < tof_search_range[-1] else len(
            tof_search_range) - 1 for i in v_list[:, 2]]
        aoa_idxs = [bisect.bisect_left(aoa_cos_search_range, i) if i < aoa_cos_search_range[-1] else len(
            aoa_cos_search_range) - 1 for i in np.cos(v_list[:, 0])]
        # target_y[tof_idxs, aoa_idxs] = attenuations
        target_y = scipy.sparse.csr_matrix((attenuations, (aoa_idxs, tof_idxs)),
                                     shape=(len(aoa_cos_search_range), len(tof_search_range)))
        res.append(target_y)
    return res
    # sigma_tof = self.resolution / self.bw / self.tof_search_step / 3
    # sigma_aoa = self.resolution * (2 / self.rx_antennas) / self.aoa_cos_search_step / 3
    # y.append(scipy.ndimage.gaussian_filter(target_y, (sigma_tof, sigma_aoa)) * 2 * np.pi * sigma_aoa * sigma_tof)


def from_path_to_2d_aoa_tof_dense_target(v_lists, tof_search_range, tof_search_step, aoa_cos_search_range, aoa_cos_search_step,
                                         resolution, bw, rx_antennas):
    res = []
    for v_list in v_lists:
        target_y = np.zeros((len(aoa_cos_search_range), len(tof_search_range)))
        attenuations = np.abs(v_list[:, 4])
        tof_idxs = [bisect.bisect_left(tof_search_range, i) if i < tof_search_range[-1] else len(
            tof_search_range) - 1 for i in v_list[:, 2]]
        aoa_idxs = [bisect.bisect_left(aoa_cos_search_range, i) if i < aoa_cos_search_range[-1] else len(
            aoa_cos_search_range) - 1 for i in np.cos(v_list[:, 0])]
        target_y[aoa_idxs, tof_idxs] = attenuations
        sigma_tof = resolution / bw / tof_search_step / 3
        sigma_aoa = resolution * (2 / rx_antennas) / aoa_cos_search_step / 3
        target_y = scipy.ndimage.gaussian_filter(target_y, (sigma_aoa, sigma_tof)) * 2 * np.pi * sigma_aoa * sigma_tof
        res.append(target_y)
    return res


class Generator_1d_tof(keras.utils.Sequence):
    def __init__(self, data, batch_size, snr_range, resolution, **kwargs):
        self.data = data
        self.batch_size = batch_size
        self.snr_range = snr_range  # tuple
        self.resolution = resolution
        self.tof_search_range = kwargs.get('tof_search_range')
        self.bw = kwargs.get('bw')
        self.tof_search_step = kwargs.get('tof_search_step')
        self.sig_generator = Signal(**kwargs)
        self.num_batches = len(data) // batch_size

    def __getitem__(self, idx):
        """
        Generate a single batch of data.
        :param item:
        :return:
        """
        x, y = [], []
        for v_list in self.data[idx * self.batch_size:(idx + 1) * self.batch_size]:
            snr = self.snr_range[0] + np.random.rand() * (self.snr_range[1] - self.snr_range[0])
            sig_f = self.sig_generator.get_new_channel_f(v_list, target_snr_db=snr)
            # leave out dc carrier and rescale st. sig_f[0,0,0] has zero phase
            sig_f_normalized = data_normalization(sig_f)
            sig_f_vectorized = data_vectorization(sig_f_normalized)
            x.append(sig_f_vectorized)

            target_y = np.zeros(len(self.tof_search_range))
            attenuations = np.abs(v_list[:, 4])
            tof_idxs = [bisect.bisect_left(self.tof_search_range, i) if i < self.tof_search_range[-1] else len(
                self.tof_search_range) - 1 for i in v_list[:, 2]]
            target_y[tof_idxs] = attenuations
            sigma_tof = self.resolution / self.bw / self.tof_search_step / 3
            y.append(scipy.ndimage.gaussian_filter1d(target_y, sigma_tof) * np.sqrt(2 * np.pi) * sigma_tof)

        return np.array(x), np.array(y)

    def __len__(self):
        """
        Number of batches per epoch.
        :return:
        """
        return self.num_batches


class Generator_2d_aoa_tof(keras.utils.Sequence):
    def __init__(self, x, y_sparse, batch_size, resolution, **kwargs):
        self.x = x
        self.y_sparse = y_sparse
        self.batch_size = batch_size
        self.resolution = resolution
        self.tof_search_range = kwargs.get('tof_search_range')
        self.aoa_cos_search_range = kwargs.get('aoa_cos_search_range')
        self.bw = kwargs.get('bw')
        self.rx_antennas = kwargs.get('rx_antennas')
        self.tof_search_step = kwargs.get('tof_search_step')
        self.aoa_cos_search_step = kwargs.get('aoa_cos_search_step')
        self.num_batches = len(x) // batch_size

    def __getitem__(self, idx):
        """
        Generate a single batch of data.
        :param item:
        :return:
        """
        x_batch = self.x[idx * self.batch_size: (idx + 1) * self.batch_size]
        y_batch = []
        sigma_tof = self.resolution / self.bw / self.tof_search_step / 3
        sigma_aoa = self.resolution * (2 / self.rx_antennas) / self.aoa_cos_search_step / 3
        for y in self.y_sparse[idx * self.batch_size: (idx + 1) * self.batch_size]:
            y_batch.append(
                scipy.ndimage.gaussian_filter(y.todense(), (sigma_aoa, sigma_tof)) * 2 * np.pi * sigma_aoa * sigma_tof)

        return x_batch, np.array(y_batch)

    def __len__(self):
        """
        Number of batches per epoch.
        :return:
        """
        return self.num_batches
