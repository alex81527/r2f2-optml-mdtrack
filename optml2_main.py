import logging
import matplotlib.pyplot as pl
from matplotlib import colors
from scipy.special import diric
from scipy import ndimage, signal
import scipy.io
import mat73
from tensorflow import keras
from optml2.core import Optml2
from optml2.data_generator import Generator_1d_tof, from_path_to_2d_aoa_tof_dense_target, \
    from_path_to_2d_aoa_tof_sparse_target, from_path_to_1d_tof_dense_target, data_normalization, \
    data_vectorization, gen_2d_aoa_tof_physical_paths
from utils.common import timer, Snr
from utils import siggen
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from mdtrack_mobicom19.core import Mdtrack
from r2f2_sigcomm16.core import R2f2
import os
import pickle
import json
import time
import gc

logger = logging.getLogger(__name__)


@timer
def train_1d_tof_resolution():
    config = {'rx_antennas': 1, 'tx_antennas': 1, 'tof_max': 300e-9,
              'nn_sparse_y': False, 'nn_data_snr_range': (10, 25), 'nn_data_size': 20000, 'nn_epochs': 1000,
              'max_physical_paths': 5}
    p = Optml2(**config)
    resolutions = [0.2, 0.25, 0.3, 0.4]
    for resolution in resolutions:
        p.nn_working_folder = f"./optml2/test_1d_tof_resolution/res{resolution:.2f}"
        p.gen_1d_tof_training_data(resolution=resolution, force=True)
        p.compile_1d_tof_model()
        p.setup_1d_tof_datagenerator(resolution=resolution)
        p.train_1d_tof_model()
        keras.backend.clear_session()
        gc.collect()


def check_1d_tof_resolution():
    """
    Sampling at 20 MHz has a base resolution of 50ns. NN can achieve roughly 3~5 times of that, i.e., 0.3 or 0.2
    of the original resolution.
    :return:
    """
    config = {'rx_antennas': 1, 'tx_antennas': 1, 'tof_max': 300e-9}
    m = Mdtrack(**config)
    p = Optml2(**config)
    # resolutions = [0.2, 0.25, 0.3, 0.4]
    resolutions = [0.3, 0.4]

    # plt, axs = pl.subplots(4, 1)
    # for i, resolution in enumerate(resolutions):
    #     history_filename = f"./optml2/test_1d_tof_resolution/res{resolution:.2f}/history.pckl"
    #     data = p.pickle_load(history_filename)
    #
    #     axs[i].plot(data['loss'])
    #     axs[i].plot(data['val_loss'])
    #     axs[i].set_title(f"res_{resolution}")
    #     axs[i].legend(['loss', 'val_loss'])

    test1_par = np.array([[np.deg2rad(110.7), np.deg2rad(0), 20.6e-9, 0.0, 1],
                          [np.deg2rad(130), np.deg2rad(0), 60.1e-9, -0.0, 1 / 5]])
    sig_t = m.get_new_channel_t(test1_par, target_snr_db=20)
    H = np.fft.fft(sig_t, axis=0) * m.ltf_f[:, np.newaxis, np.newaxis]
    sig_f_normalized = data_normalization(H)
    sig_f_vectorized = data_vectorization(sig_f_normalized)

    plt2, axs2 = pl.subplots(max(2, len(resolutions)), 3)
    for i, resolution in enumerate(resolutions):
        model = keras.models.load_model(f"./optml2/test_1d_tof_resolution/res{resolution:.2f}/model.h5")

        # # view validation data
        # x_train = p.pickle_load(f"./optml2/test_1d_tof_resolution/res{resolution:.2f}/x_training.pckl")
        # x_valid = p.pickle_load(f"./optml2/test_1d_tof_resolution/res{resolution:.2f}/x_validation.pckl")
        # x_train_raw = p.pickle_load(f"./optml2/test_1d_tof_resolution/res{resolution:.2f}/x_training_raw_paths.pckl")
        # x_valid_raw = p.pickle_load(f"./optml2/test_1d_tof_resolution/res{resolution:.2f}/x_validation_raw_paths.pckl")
        # args = (x_valid_raw[105:108], p.tof_search_range, p.tof_search_step, resolution, p.bw)
        # y_valid = np.array(from_path_to_1d_tof_dense_target(*args))
        # predicted_valid = model(x_valid[105:108])
        # for j in range(3):
        #     axs2[i, j].plot(p.tof_search_range * 1e9, y_valid[j, :])
        #     axs2[i, j].plot(p.tof_search_range * 1e9, predicted_valid[j, :])
        #     axs2[i, j].set_title(f"res_{resolution}, data{j}")
        #     print(
        #         f"res_{resolution}, data{j}, y_valid{signal.find_peaks(y_valid[j, :])}, predicted{signal.find_peaks(predicted_valid[j, :], prominence=(0.1, None))}")
        #     # axs2[i, j].legend(['expected', 'predicted'])

        # view simulated data
        predicted_valid = model(np.array([sig_f_vectorized]))
        axs2[i, 0].plot(p.tof_search_range * 1e9, predicted_valid[0, :], 'g')
        axs2[i, 0].stem(test1_par[:, 2] * 1e9, np.abs(test1_par[:, 4]))
        axs2[i, 0].set_title(f"res_{resolution}, data{0}")
    pl.show()


def train_2d_aoa_tof_resolution():
    # config = {'rx_antennas': 3, 'tx_antennas': 1, 'fc': 5310e6, 'bw': 40e6,
    #           'tof_max': 300e-9, 'tof_search_step': 5e-9, 'aoa_cos_search_step': 0.05,
    #           'nn_sparse_y': True, 'nn_data_snr_range': (3, 20), 'nn_data_size': 2000000, 'nn_epochs': 1000,
    #           'max_physical_paths': 4}
    rx_ant, tx_ant, fftsize, amin = 4, 1, 64, 0.1
    resolutions = [1.0]
    for resolution in resolutions:
        for L in [6]:
            for H in [1]:
                config = {'rx_antennas': rx_ant, 'tx_antennas': tx_ant, 'fc': 5310e6, 'bw': 40e6,
                          'tof_max': 250e-9, 'tof_search_step': 5e-9, 'aoa_cos_search_step': 0.05,
                          'nn_sparse_y': True, 'nn_data_snr_range': (20, 30), 'nn_data_size': 200000, 'nn_epochs': 1000,
                          'nn_layers': L, 'nn_unit_per_layer': H * 2 * (fftsize - 1) * rx_ant * tx_ant,
                          'max_physical_paths': 5}
                p = Optml2(**config)

                p.nn_working_folder = f"./optml2/test_2d_aoa_tof_resolution/{rx_ant}rx/res{resolution:.2f}/L{L}H{H}_amin{amin}"
                p.gen_2d_aoa_tof_training_data(resolution=resolution, force=True)

                aoas, tofs = [], []
                x_validation_raw_paths = p.pickle_load(f"{p.nn_working_folder}/x_validation_raw_paths.pckl")
                for x_valid in x_validation_raw_paths:
                    # print(np.rad2deg(np.real(x_valid[:,0])), 1e9*np.real(x_valid[:,2]))
                    aoas.extend(np.rad2deg(np.real(x_valid[:, 0])))
                    tofs.extend(1e9 * np.real(x_valid[:, 2]))
                pl.figure()
                pl.hist2d(tofs, aoas, bins=100, range=[[0, 300], [0, 180]], norm=colors.LogNorm())
                pl.colorbar()
                pl.xlabel('ToF (ns)')
                pl.ylabel('AoA (deg)')
                pl.show()
                # exit()

                p.compile_2d_aoa_tof_model()
                p.setup_2d_aoa_tof_datagenerator(resolution=resolution)
                p.train_2d_aoa_tof_model()
                keras.backend.clear_session()
                gc.collect()


def check_2d_aoa_tof_resolution():
    """
    Sampling at 20 MHz has a base resolution of 50ns. NN can achieve roughly 3~5 times of that, i.e., 0.3 or 0.2
    of the original resolution.
    :return:
    """

    m = Mdtrack()
    rx_ant, tx_ant, fftsize, amin = 4, 1, 64, 0.0
    resolutions = [1.0]
    for resolution in resolutions:
        for L in [6]:
            for H in [1]:
                config = {'rx_antennas': rx_ant, 'tx_antennas': tx_ant, 'fc': 5310e6, 'bw': 40e6,
                          'tof_max': 250e-9, 'tof_search_step': 5e-9, 'aoa_cos_search_step': 0.05,
                          'nn_sparse_y': True, 'nn_data_snr_range': (20, 30), 'nn_data_size': 100000, 'nn_epochs': 1000,
                          'nn_layers': L, 'nn_unit_per_layer': H * 2 * (fftsize - 1) * rx_ant * tx_ant,
                          'max_physical_paths': 5}
                p = Optml2(**config)
                folder_prefix = f"./optml2/test_2d_aoa_tof_resolution/{rx_ant}rx/res{resolution:.2f}/L{L}H{H}_amin{amin}"

                params = p.pickle_load(f"{folder_prefix}/parameters.pckl")
                x_validation_raw_paths = p.pickle_load(f"{folder_prefix}/x_validation_raw_paths.pckl")
                history = p.pickle_load(f"{folder_prefix}/history.pckl")
                print(f"resolution={resolution}, {params}")

                # # plot training data distribution
                # aoas, tofs = [], []
                # for x_valid in x_validation_raw_paths[:100]:
                #     # print(np.rad2deg(np.real(x_valid[:,0])), 1e9*np.real(x_valid[:,2]))
                #     aoas.extend(np.rad2deg(np.real(x_valid[:, 0])))
                #     tofs.extend(1e9 * np.real(x_valid[:, 2]))
                # pl.figure()
                # pl.hist2d(tofs, aoas, bins=100, range=[[0, 300], [0, 180]], norm=colors.LogNorm())
                # pl.colorbar()
                # pl.xlabel('ToF (ns)')
                # pl.ylabel('AoA (deg)')
                # pl.title(f"res{resolution:.2f}/L{L}H{H}")
                # pl.show()
                # # exit()

                # plot history
                pl.figure()
                pl.plot(history['loss'])
                pl.plot(history['val_loss'])
                pl.legend(['loss', 'val_loss'])
                pl.title(f"{rx_ant}rx/res{resolution:.2f}/L{L}H{H}_amin{amin}")

                # show a few validation samples
                tof_min, tof_max, tof_step = eval(params['tof_search_range (nanosecond)'])
                tof_ns = np.arange(tof_min, tof_max, tof_step)
                cos_min, cos_max, cos_step = eval(params['aoa_cos_search_range (cos)'])
                aoa_deg = np.rad2deg(np.arccos(np.arange(cos_min, cos_max, cos_step)))

                plt, axs = pl.subplots(4, 2)
                model = keras.models.load_model(f"{folder_prefix}/model.h5")
                x_train = p.pickle_load(f"{folder_prefix}/x_training.pckl")
                x_valid = p.pickle_load(f"{folder_prefix}/x_validation.pckl")
                x_train_raw = p.pickle_load(f"{folder_prefix}/x_training_raw_paths.pckl")
                x_valid_raw = p.pickle_load(f"{folder_prefix}/x_validation_raw_paths.pckl")
                data_idx = np.random.choice(np.arange(500), 4)
                args = (x_valid_raw[data_idx], p.tof_search_range, p.tof_search_step, p.aoa_cos_search_range,
                        p.aoa_cos_search_step, resolution, p.bw, p.rx_antennas)
                y_valid = np.array(from_path_to_2d_aoa_tof_dense_target(*args))
                predicted_y = model(x_valid[data_idx, :])

                for j in range(len(data_idx)):
                    axs[j, 0].imshow(y_valid[j], aspect='auto', cmap=pl.get_cmap('jet'),
                                     extent=[tof_ns[0], tof_ns[-1], aoa_deg[-1], aoa_deg[0]])
                    if j == 0:
                        axs[j, 0].set_ylabel('AoA (Deg)')
                        axs[j, 0].set_xlabel('ToF (ns)')
                    axs[j, 1].imshow(predicted_y[j], aspect='auto', cmap=pl.get_cmap('jet'),
                                     extent=[tof_ns[0], tof_ns[-1], aoa_deg[-1], aoa_deg[0]])
                    print(m.print_pretty(x_valid_raw[data_idx[j]]))

                pl.show()


# plt, axs = pl.subplots(2, 2)
# pl.figure()
# model = keras.models.load_model(f"{folder_prefix}/res1.00/model.h5")
# x_valid = p.pickle_load(f"{folder_prefix}/res0.70/x_validation.pckl")
# x_valid_raw = p.pickle_load(
#     f"{folder_prefix}/res0.70/x_validation_raw_paths.pckl")
# args = (x_valid_raw[0:10], p.tof_search_range, p.tof_search_step, p.aoa_cos_search_range,
#         p.aoa_cos_search_step, 0.7, p.bw, p.rx_antennas)
# y_valid = np.array(from_path_to_2d_aoa_tof_dense_target(*args))
# predicted_y = model(x_valid[0:10, :])
#
# # pl.imshow(y_valid[6], aspect='auto', cmap=pl.get_cmap('jet'))
# pl.imshow(predicted_y[6], aspect='auto', cmap=pl.get_cmap('jet'))
# pl.show()


def test_on_experiment_data():
    def see_doppler_at_each_antenna(m, mainfolder, subfolder, start_index=1):
        # see doppler effects at each antenna
        preamble_train = []
        for i in range(start_index, start_index + m.preamble_repeat_cnt):
            lts_samples = scipy.io.loadmat(f"{mainfolder}/{subfolder}/lts_samples_{i}.mat")['lts_samples']
            # H_t = np.fft.ifft(H*m.ltf_f[:, np.newaxis], axis=0)
            preamble_train.append(lts_samples[:, :, np.newaxis])

        preamble_train = np.array(preamble_train)
        plt, axs = pl.subplots(int(np.ceil(m.rx_antennas / 2)), 2)
        for i in range(m.rx_antennas):
            cor = [np.abs(np.vdot(x, preamble_train[:, :, i, 0])) for x in m.dop_search_mat]
            axs[i // 2, i % 2].plot(10 * np.log10(cor))
            # axs[i // 2, i % 2].plot(cor)
            xtick_pos = np.arange(0, len(m.dop_search_range), len(m.dop_search_range) // 5)
            axs[i // 2, i % 2].set_xticks(xtick_pos)
            axs[i // 2, i % 2].set_xticklabels([str(x) for x in m.dop_search_range[xtick_pos]])
            axs[i // 2, i % 2].set_title(f"rx_antenna {i + 1}")
            axs[i // 2, i % 2].set_xlabel(f"doppler shift (Hz)")
            axs[i // 2, i % 2].set_ylabel(f"Mag (dB)")
        pl.show()
        # plt.savefig(f"./matlab_scripts/experiment_data/{subfolder}.pdf")

    def get_aoa_doppler_map(m, mainfolder, subfolder, start_index=1):
        # see doppler effects at each antenna
        preamble_train = []
        for i in range(start_index, start_index + m.preamble_repeat_cnt):
            lts_samples = scipy.io.loadmat(f"{mainfolder}/{subfolder}/lts_samples_{i}.mat")['lts_samples']
            # H_t = np.fft.ifft(H*m.ltf_f[:, np.newaxis], axis=0)
            preamble_train.append(lts_samples[:, :, np.newaxis])

        preamble_train = np.array(preamble_train)

        H = np.fft.fft(preamble_train, axis=1) * m.ltf_f[np.newaxis, :, np.newaxis, np.newaxis]

        admap = np.zeros((len(m.dop_search_range), len(m.aoa_search_range)))
        for i, dop_mat in enumerate(m.dop_search_mat):
            for j, aoa_mat in enumerate(m.aoa_search_mat):
                H_p = np.sum(H * aoa_mat.conjugate()[:, :, :, np.newaxis], axis=2)
                H_pp = np.sum(H_p, axis=2)
                y_pp = np.fft.ifft(H_pp * m.ltf_f[np.newaxis, :], axis=1)
                zval = np.vdot(dop_mat, y_pp)
                admap[i, j] = abs(zval)

        pl.figure()
        pl.imshow(10 * np.log10(admap), aspect='auto', cmap=pl.get_cmap('jet'))
        pl.xticks(np.arange(0, len(m.aoa_search_range), len(m.aoa_search_range) // 10),
                  np.round(np.rad2deg(m.aoa_search_range[
                                          np.arange(0, len(m.aoa_search_range), len(m.aoa_search_range) // 10)]), 1))
        pl.yticks(np.arange(0, len(m.dop_search_range), len(m.dop_search_range) // 10),
                  np.round(m.dop_search_range[
                               np.arange(0, len(m.dop_search_range),
                                         len(m.dop_search_range) // 10)], 1))
        pl.xlabel('AoA (deg)')
        pl.ylabel('Dop (Hz)')
        pl.colorbar()
        pl.show()
        # plt.savefig(f"./matlab_scripts/experiment_data/{subfolder}.pdf")

    def load_lts_train(m, mainfolder, subfolder, tof_shift_second, start_index):
        preamble_train = []
        rf_calibration = scipy.io.loadmat(f"{mainfolder}/{subfolder}/rf_calibration.mat")['rf_calibration']
        for i in range(start_index, start_index + m.preamble_repeat_cnt):
            lts_samples = scipy.io.loadmat(f"{mainfolder}/{subfolder}/lts_samples_{i}.mat")['lts_samples']
            H = np.fft.fft(lts_samples, axis=0) * rf_calibration
            H = H * m.get_tof_effect(tof_shift_second)[:, np.newaxis]
            H_t = np.fft.ifft(H, axis=0)
            # LOS = m.get_new_channel_t([[np.deg2rad(137.51), 0, 85.5e-9, 0, 4.59*np.exp(1j*-0.23)]])
            # H_t = H_t - LOS[:,:,0]
            preamble_train.append(H_t[:, :, np.newaxis])

        return np.array(preamble_train)

    def load_lts_train2(m, mainfolder, subfolder, tof_shift_second, preamble_cnt):
        preamble_train = []
        rf_calibration = mat73.loadmat(f"{mainfolder}/{subfolder}/rf_calibration.mat")['rf_calibration']
        lts_samples = mat73.loadmat(f"{mainfolder}/{subfolder}/lts_sample_train.mat")['lts_sample_train'].T
        assert m.bw % (64*preamble_cnt) == 0
        interval = int(m.bw / 64 / preamble_cnt)
        for i in range(0, int(m.bw/64), interval):
            H = np.fft.fft(lts_samples[i*64:(i+1)*64, :], axis=0) * rf_calibration
            H = H * m.get_tof_effect(tof_shift_second)[:, np.newaxis]
            H_t = np.fft.ifft(H, axis=0)
            preamble_train.append(H_t[:, :, np.newaxis])

        return np.array(preamble_train)

    @timer
    def run_optml(r, p, H, model_2d):
        r.channel_est = H
        x = np.array([data_vectorization(data_normalization(H))])
        predicted_y = model_2d(x)
        hmap = predicted_y[0].numpy()
        peaks = p.filter_peaks(p.get_ranked_peaks(hmap, debug=True))
        p.show_aoa_tof_heatmap(hmap, peaks)
        res = r.solve_optimization(peaks.ravel())
        return r.get_physical_path_parameters(res)

    @timer
    def run_r2f2(r, p, H):
        r.channel_est = H
        hmap = r.get_aoa_tof_heatmap(H[:, :, 0])
        peaks = r.filter_peaks(r.get_ranked_peaks(hmap, debug=True))
        r.show_aoa_tof_heatmap(hmap, peaks)
        res = r.solve_optimization(peaks.ravel())
        return r.get_physical_path_parameters(res)

    @timer
    def run_mdtrack(m, sig_t):
        return m.resolve_multipath(sig_t)

    def plot_channel(m, sig_t_orig, v_list):
        sig_t_est = m.get_new_channel_t(v_list)
        plt, axs = pl.subplots(2, m.rx_antennas)
        axs = axs.reshape(2, m.rx_antennas)
        for i in range(m.rx_antennas):
            ch1 = np.fft.fft(sig_t_orig[:, i, 0]) * m.ltf_f
            ch2 = np.fft.fft(sig_t_est[:, i, 0]) * m.ltf_f

            ch1 = np.fft.fftshift(ch1)
            ch1 = ch1[np.abs(ch1) > 1e-12]
            ch2 = np.fft.fftshift(ch2)
            ch2 = ch2[np.abs(ch2) > 1e-12]

            axs[0, i].plot(10 * np.log10(np.abs(ch1)))
            axs[0, i].plot(10 * np.log10(np.abs(ch2)))
            axs[0, i].set_xlabel('subcarriers')
            axs[0, i].set_ylabel('Magnitude (dB)')
            axs[0, i].set_title(f"antenna {i + 1}")
            axs[1, i].plot(np.angle(ch1))
            axs[1, i].plot(np.angle(ch2))
            axs[1, i].set_xlabel('subcarriers')
            axs[1, i].set_ylabel('Phase (radian)')
            axs[1, i].set_title(f"antenna {i + 1}")

            if i == 0:
                axs[0, i].legend(['measured', 'fitted'])
                # axs[1, i].legend(['measured', 'fitted'])
        pl.show()

    mainfolder = 'C:/Users/alex/Desktop/WARPLab_Reference_Design_7.7.1/M_Code_Examples/matlab_scripts/experiment_data'
    # mainfolder = './matlab_scripts/experiment_data'
    # subfolder = 'preamble_train_antenna_static_5310e6_40e6'
    # subfolder = 'preamble_train_human_walking_close_5310e6_40e6'
    # subfolder = 'preamble_train_human_walking_away_5310e6_40e6'
    # subfolder = 'lts_sample_train_walk_close_5310e6_40e6'
    subfolder = ''

    preamble_repeat_cnt = 1
    rx_ant = 3
    amin, L, H = 0.2, 3, 1
    resolution = 1.0
    fc = 5310e6
    bw = 20e6
    noise_pwr = scipy.io.loadmat(f"{mainfolder}/{subfolder}/adjusted_noisepwr_{1}.mat")['adjusted_noisepwr'][0]
    # noise_pwr = mat73.loadmat(f"{mainfolder}/{subfolder}/noise_pwr.mat")['noise_pwr']
    md_config = {'rx_antennas': rx_ant, 'tx_antennas': 1, 'tof_max': 64/bw, 'aod_max': 0.02,
                 'initial_est_stop_threshold_db': 3.0, 'debug': True, 'per_tone_noise_mw': 0.0035,
                 'rf_chain_noise_pwr': noise_pwr, 'fc': fc, 'bw': bw,
                 'dop_max': preamble_repeat_cnt / 2, 'dop_search_step': 1,
                 'preamble_interval_sec': 1 / preamble_repeat_cnt,
                 'preamble_repeat_cnt': preamble_repeat_cnt}
    m = Mdtrack(**md_config)
    config = {'rx_antennas': rx_ant, 'tx_antennas': 1, 'fc': fc, 'bw': bw,
              'tof_max': 250e-9, 'tof_search_step': 5e-9, 'aoa_cos_search_step': 0.05,
              'nn_sparse_y': True, 'nn_data_snr_range': (15, 25), 'nn_data_size': 100000, 'nn_epochs': 1000,
              'max_physical_paths': 5}
    p = Optml2(**config)

    config = {'rx_antennas': rx_ant, 'tx_antennas': 1, 'fc': fc, 'bw': bw,
              'tof_max': 250e-9, 'tof_search_step': 0.5e-9,
              'aoa_search_step': 0.01, 'dop_max': 0.1, 'debug': True}
    r = R2f2(**config)

    # model_2d = keras.models.load_model(
    #     f"./optml2/test_2d_aoa_tof_resolution/{rx_ant}rx/res{resolution:.2f}/L{L}H{H}_amin{amin}/model.h5")
    # model_1d = keras.models.load_model(f"./optml2/test_1d_tof_resolution/res0.40/model.h5")
    # print(p.pickle_load(f"./optml2/test_1d_tof_resolution/res0.40/parameters.pckl"))


    # see_doppler_at_each_antenna(m, mainfolder, subfolder, start_index=1)
    # get_aoa_doppler_map(m, mainfolder, subfolder, start_index=1)
    # exit()

    # load data
    tof_shift_second = 0/bw
    start_index = 1
    preamble_train = load_lts_train(m, mainfolder, subfolder, tof_shift_second, start_index)
    # preamble_train = load_lts_train2(m, mainfolder, subfolder, tof_shift_second, preamble_cnt=m.preamble_repeat_cnt)
    # scipy.io.savemat(f"{mainfolder}/{subfolder}/processed_lts_sample_train.mat", {'processed_lts_sample_train': preamble_train})
    # preamble_train = scipy.io.loadmat(f"{mainfolder}/{subfolder}/processed_lts_sample_train.mat")['processed_lts_sample_train']
    print(preamble_train.shape)

    # get channel
    H = np.fft.fft(preamble_train, axis=1) * m.ltf_f[np.newaxis, :, np.newaxis, np.newaxis]
    # exit()

    # # optml
    # v_list, runtime = run_optml(r, p, H[0,:,:,:], model_2d)
    # plot_channel(m, preamble_train[0,:,:,:], v_list)
    # t2 = m.get_new_channel_t(v_list, target_snr_db=None)
    # f2 = np.fft.fft(t2, axis=0) * m.ltf_f[:, np.newaxis, np.newaxis]
    # for i in range(m.preamble_repeat_cnt):
    #     v_list, runtime = run_optml(r, p, H[i,:,:,:] - f2, model_2d)
    #     # plot_channel(m, preamble_train[0, :, :, :] - t2, v_list)
    #     print(f"ml {i + 1}-th", m.print_pretty(v_list))

    # # r2f2
    # v_list, runtime = run_r2f2(r, p, H[0, :, :, :])
    v_list = r.resolve_multipath(preamble_train[0,:,:,:])
    print('r2f2 1st', m.print_pretty(v_list))
    plot_channel(m, preamble_train[0, :, :, :], v_list)
    #
    # t2 = m.get_new_channel_t(v_list, target_snr_db=None)
    # f2 = np.fft.fft(t2, axis=0) * m.ltf_f[:, np.newaxis, np.newaxis]
    # for i in range(m.preamble_repeat_cnt):
    #     v_list, runtime = run_r2f2(r, p, H[i, :, :, :] - f2)
    #     # plot_channel(m, preamble_train[0, :, :, :] - t2, v_list)
    #     print(f"r2f2 {i+1}-th", m.print_pretty(v_list))
    # exit()

    # mdtrack
    v_list, runtime = run_mdtrack(m, preamble_train)
    plot_channel(m, preamble_train[0,:,:,:], v_list)

    # for i in range(2):
    #     res, runtime = run_optml(r, p, H[0, :, :, :], model_2d)
    #     v_list = r.get_physical_path_parameters(res)
    #     print('ml', m.print_pretty(v_list))
    #     H[0, :, :, :] = np.fft.fft(preamble_train[0,:,:,:] - m.get_new_channel_t(v_list), axis=0) * m.ltf_f[:, np.newaxis, np.newaxis]
    # plot_channel(m, preamble_train[0, :, :, :], v_list)
    # exit()


def test_on_sim_data():
    @timer
    def run_mdtrack(m, sig_t):
        return m.resolve_multipath(sig_t)

    @timer
    def run_optml(r, p, H, model_2d):
        r.channel_est = H
        x = np.array([data_vectorization(data_normalization(H))])
        predicted_y = model_2d(x)
        hmap = predicted_y[0].numpy()
        peaks = p.filter_peaks(p.get_ranked_peaks(hmap, debug=True))
        p.show_aoa_tof_heatmap(hmap, peaks)
        res = r.solve_optimization(peaks.ravel())
        return r.get_physical_path_parameters(res)

    @timer
    def run_r2f2(r, p, H):
        r.channel_est = H
        hmap = r.get_aoa_tof_heatmap(H[:, :, 0])
        peaks = r.filter_peaks(r.get_ranked_peaks(hmap, debug=True))
        r.show_aoa_tof_heatmap(hmap, peaks)
        res = r.solve_optimization(peaks.ravel())
        return r.get_physical_path_parameters(res)

    def plot_channel(m, sig_t_orig, v_list):
        sig_t_est = m.get_new_channel_t(v_list)
        plt, axs = pl.subplots(2, m.rx_antennas)
        axs = axs.reshape(2, m.rx_antennas)
        for i in range(m.rx_antennas):
            ch1 = np.fft.fft(sig_t_orig[:, i, 0]) * m.ltf_f
            ch2 = np.fft.fft(sig_t_est[:, i, 0]) * m.ltf_f

            ch1 = np.fft.fftshift(ch1)
            ch1 = ch1[np.abs(ch1) > 1e-12]
            ch2 = np.fft.fftshift(ch2)
            ch2 = ch2[np.abs(ch2) > 1e-12]

            axs[0, i].plot(10 * np.log10(np.abs(ch1)))
            axs[0, i].plot(10 * np.log10(np.abs(ch2)))
            axs[0, i].set_xlabel('subcarriers')
            axs[0, i].set_ylabel('Magnitude (dB)')
            axs[0, i].set_title(f"antenna {i + 1}")
            axs[1, i].plot(np.angle(ch1))
            axs[1, i].plot(np.angle(ch2))
            axs[1, i].set_xlabel('subcarriers')
            axs[1, i].set_ylabel('Phase (radian)')
            axs[1, i].set_title(f"antenna {i + 1}")

            if i == 0:
                axs[0, i].legend(['measured', 'fitted'])
                # axs[1, i].legend(['measured', 'fitted'])
        pl.show()

    def get_aoa_doppler_map(m, preamble_train):
        # see doppler effects at each antenna
        # preamble_train = []
        # for i in range(start_index, start_index + m.preamble_repeat_cnt):
        #     lts_samples = scipy.io.loadmat(f"{mainfolder}/{subfolder}/lts_samples_{i}.mat")['lts_samples']
        #     # H_t = np.fft.ifft(H*m.ltf_f[:, np.newaxis], axis=0)
        #     preamble_train.append(lts_samples[:, :, np.newaxis])
        #
        # preamble_train = np.array(preamble_train)

        H = np.fft.fft(preamble_train, axis=1) * m.ltf_f[np.newaxis, :, np.newaxis, np.newaxis]

        admap = np.zeros((len(m.dop_search_range), len(m.aoa_search_range)))
        for i, dop_mat in enumerate(m.dop_search_mat):
            for j, aoa_mat in enumerate(m.aoa_search_mat):
                H_p = np.sum(H * aoa_mat.conjugate()[:, :, :, np.newaxis], axis=2)
                H_pp = np.sum(H_p, axis=2)
                y_pp = np.fft.ifft(H_pp * m.ltf_f[np.newaxis, :], axis=1)
                zval = np.vdot(dop_mat, y_pp)
                admap[i, j] = abs(zval)

        pl.figure()
        pl.imshow(10 * np.log10(admap / admap.max()), aspect='auto', cmap=pl.get_cmap('jet'))
        pl.xticks(np.arange(0, len(m.aoa_search_range), len(m.aoa_search_range) // 10),
                  np.round(np.rad2deg(m.aoa_search_range[
                                          np.arange(0, len(m.aoa_search_range), len(m.aoa_search_range) // 10)]), 1))
        pl.yticks(np.arange(0, len(m.dop_search_range), len(m.dop_search_range) // 10),
                  np.round(m.dop_search_range[
                               np.arange(0, len(m.dop_search_range),
                                         len(m.dop_search_range) // 10)], 1))
        pl.xlabel('AoA (deg)')
        pl.ylabel('Dop (Hz)')
        pl.colorbar()
        pl.show()
        # plt.savefig(f"./matlab_scripts/experiment_data/{subfolder}.pdf")

    def beamforming_gain_db(true_H, predicted_H):
        original_pwr = np.mean(np.abs(np.sum(true_H, axis=1)) ** 2)
        beamformed_pwr = np.mean(np.abs(np.sum(true_H * np.exp(-1j * np.angle(predicted_H)), axis=1)) ** 2)
        gain = beamformed_pwr / original_pwr
        return 10 * np.log10(gain)

    preamble_repeat_cnt = 40
    rx_ant = 4
    amin, L, H = 0.0, 6, 1
    resolution = 1.0
    fc = 5310e6
    bw = 40e6
    # mainfolder = './matlab_scripts/experiment_data'
    # subfolder = 'lts_sample_train_walk_close_5310e6_40e6'
    # # noise_pwr = scipy.io.loadmat(f"{mainfolder}/{subfolder}/adjusted_noisepwr_{1}.mat")['adjusted_noisepwr'][0]
    # noise_pwr = mat73.loadmat(f"{mainfolder}/{subfolder}/noise_pwr.mat")['noise_pwr']
    md_config = {'rx_antennas': rx_ant, 'tx_antennas': 1, 'tof_max': 64 / bw, 'aod_max': 0.02,
                 'initial_est_stop_threshold_db': 3.0, 'debug': True,
                 'fc': fc, 'bw': bw,
                 'dop_max': preamble_repeat_cnt / 2, 'dop_search_step': 1,
                 'preamble_interval_sec': 1 / preamble_repeat_cnt,
                 'preamble_repeat_cnt': preamble_repeat_cnt}
    m = Mdtrack(**md_config)
    config = {'rx_antennas': rx_ant, 'tx_antennas': 1, 'fc': fc, 'bw': bw,
              'tof_max': 250e-9, 'tof_search_step': 5e-9, 'aoa_cos_search_step': 0.05,
              'nn_sparse_y': True, 'nn_data_snr_range': (15, 25), 'nn_data_size': 100000, 'nn_epochs': 1000,
              'max_physical_paths': 5}
    p = Optml2(**config)

    config = {'rx_antennas': rx_ant, 'tx_antennas': 1, 'fc': fc, 'bw': bw,
              'tof_max': 250e-9, 'tof_search_step': 0.5e-9,
              'aoa_search_step': 0.01, 'dop_max': 0.1, 'debug': True}
    r = R2f2(**config)

    folder_prefix = f"./optml2/test_2d_aoa_tof_resolution/{rx_ant}rx/res{resolution:.2f}/L{L}H{H}_amin{amin}"

    # gen channel
    # x_valid_raw_list = p.pickle_load(f"{folder_prefix}/x_validation_raw_paths.pckl")
    # test1_par = x_valid_raw_list[5]
    test1_par = np.array([[np.deg2rad(85.7), np.deg2rad(0), 17.6e-9, 0.0, 0.8],
                          [np.deg2rad(120), np.deg2rad(0), 45.1e-9, 0.0, 0.8 / 10]])
    sig_t = m.get_new_preamble_train_t(test1_par, target_snr_db=27)
    print('preamble shape', sig_t.shape)
    H = np.fft.fft(sig_t, axis=1) * m.ltf_f[np.newaxis, :, np.newaxis, np.newaxis]
    print('ground truth', m.print_pretty(test1_par))


    # # optml
    # # print(p.pickle_load(f"{folder_prefix}/parameters.pckl"))
    # model_2d = keras.models.load_model(f"{folder_prefix}/model.h5")
    # v_list = r.resolve_multipath_ml(sig_t[0, :, :, :], p, model_2d)
    # # v_list, runtime = run_optml(r, p, H[0, :, :, :], model_2d)
    # plot_channel(m, sig_t[0, :, :, :], v_list)
    # print('ml 1st', m.print_pretty(v_list))
    #
    # t2 = m.get_new_channel_t(v_list, target_snr_db=None)
    # f2 = np.fft.fft(t2, axis=0) * m.ltf_f[:, np.newaxis, np.newaxis]
    # v_list, runtime = run_optml(r, p, H[0, :, :, :] - f2, model_2d)
    # plot_channel(m, sig_t[0, :, :, :] - t2, v_list)
    # print('ml 2nd', m.print_pretty(v_list))
    # exit()

    # r2f2
    # v_list, runtime = run_r2f2(r, p, H[0, :, :, :])
    r.channel_est = H[0, :, :, :]
    # r.show_aoa_tof_cost_func()
    print('ground truth cost',r.objective_func(np.ravel([ [x,y] for x,y in zip(test1_par[:,0], test1_par[:,2])]), [1e5, 1e5]))

    res = r.solve_optimization(np.ravel([[x, y] for x, y in zip(test1_par[:, 0], test1_par[:, 2])]))
    v_list = r.get_physical_path_parameters(res)
    print('starting optimization from ground truth solution',r.print_pretty(v_list))

    v_list = r.resolve_multipath(sig_t[0, :, :, :])
    plot_channel(m, sig_t[0, :, :, :], v_list)
    print('r2f2 1st', m.print_pretty(v_list))
    exit()

    t2 = m.get_new_channel_t(v_list, target_snr_db=None)
    f2 = np.fft.fft(t2, axis=0) * m.ltf_f[:, np.newaxis, np.newaxis]
    v_list, runtime = run_r2f2(r, p, H[0, :, :, :] - f2)
    plot_channel(m, sig_t[0, :, :, :] - t2, v_list)
    print('r2f2 2nd', m.print_pretty(v_list))
    # exit()

    # mdtrack
    v_list, runtime = run_mdtrack(m, sig_t)
    plot_channel(m, sig_t[0, :, :, :], v_list)
    print('mdtrack', m.print_pretty(v_list))
    exit()

    # # plot
    # sim_result = p.pickle_load('./sim_result.pckl')
    # plt, axs = pl.subplots(2, 1)
    # axs[0].plot(np.sort(sim_result['runtime_mdtrack']),
    #             np.arange(1, 1+len(sim_result['runtime_mdtrack'])) / len(sim_result['runtime_mdtrack']),
    #                                                                    label='runtime_mdtrack')
    # axs[0].plot(np.sort(sim_result['runtime_optml']),
    #             np.arange(1, 1+len(sim_result['runtime_optml'])) / len(sim_result['runtime_optml']),
    #                                                                    label='runtime_optml')
    # axs[0].legend()
    # axs[1].plot(np.sort(sim_result['bg_mdtrack']),
    #             np.arange(1, 1+len(sim_result['bg_mdtrack'])) / len(sim_result['bg_mdtrack']),
    #                                                                    label='bg_mdtrack')
    # axs[1].plot(np.sort(sim_result['bg_optml']),
    #             np.arange(1, 1+len(sim_result['bg_optml'])) / len(sim_result['bg_optml']),
    #                                                               label='bg_optml')
    # axs[1].plot(np.sort(sim_result['bg_oracle']),
    #             np.arange(1, 1+len(sim_result['bg_oracle'])) / len(sim_result['bg_oracle']),
    #                                                               label='bg_oracle')
    # axs[1].legend()
    # pl.show()
    # exit()

    test1_par = np.array([[np.deg2rad(70.7), np.deg2rad(0), 17.6e-9, 0.0, 0.8],
                          [np.deg2rad(120), np.deg2rad(0), 65.1e-9, 0.0, 0.8/10]])
    # test1_par = gen_2d_aoa_tof_physical_paths(1, resolution, bw, p.tof_max, p.rx_antennas, 4)[0]
    # ml_par = np.array([[np.deg2rad(70.7), np.deg2rad(0), 17.1e-9, 0.0, 0*np.exp(1j*2.27)]])

    sim_result = {'runtime_mdtrack': [], 'runtime_optml': [], 'bg_mdtrack': [], 'bg_optml': [], 'bg_oracle': []}
    methods = set(['ml', 'fft'])
    sig_t = m.get_new_channel_t(test1_par, target_snr_db=25)
    print('ground truth', m.print_pretty(test1_par))
    for x_valid_raw in x_valid_raw_list[:1]:
        # preamble_train = m.get_new_preamble_train_t(test1_par, target_snr_db=20)
        # get_aoa_doppler_map(m, preamble_train)
        # exit()

        # test1_par = x_valid_raw
        # sig_t = m.get_new_channel_t(test1_par, target_snr_db=25)
        sig_f = np.fft.fft(sig_t, axis=0)
        H = sig_f * m.ltf_f[:, np.newaxis, np.newaxis]
        print(f"sig power (db) = {Snr.avg_rf_chain_snr_db(sig_t[np.newaxis,:,:,:], 1e-5)}")
        # scipy.io.savemat('matlab_scripts/calibrated_H.mat', {'calibrated_H': H[:, :, 0]})
        # H = scipy.io.loadmat('matlab_scripts/calibrated_H.mat')['calibrated_H'][:,:,np.newaxis]
        # exit()
        # m.show_freq_response(sig_t)

        if 'ml' in methods:
            # load model and prevent cold start
            model_2d = keras.models.load_model(f"{folder_prefix}/model.h5")
            x = np.array([data_vectorization(data_normalization(np.random.rand(p.fftsize, p.rx_antennas, p.tx_antennas)))])
            predicted_y = model_2d(x)

            x = np.array([data_vectorization(data_normalization(H))])
            predicted_y = model_2d(x)
            hmap = predicted_y[0].numpy()
            peaks = p.filter_peaks(p.get_ranked_peaks(hmap, debug=True))
            p.show_aoa_tof_heatmap(hmap, peaks)
            # exit()
            # ml_peak = get_ranked_peaks(p, p)
            # ml_peak = filter_peaks(ml_peak, p)
            # plt, axs = pl.subplots(1, 2)
            # axs[0].imshow(predicted_y[0], aspect='auto', cmap=pl.get_cmap('jet'))
            # for aoa_rad, tof in ml_peak:
            #     axs[0].annotate(f"x", xy=(np.where(np.arccos(p.aoa_cos_search_range)==aoa_rad)[0][0], np.where(p.tof_search_range==tof)[0][0]),
            #                     fontsize='large', color='w')
            # axs[0].set_title('ml predicted')
            #
            # ideal_sparse = from_path_to_2d_aoa_tof_sparse_target([test1_par], p.tof_search_range, p.tof_search_step,
            #                                                      p.aoa_cos_search_range, p.aoa_cos_search_step)[0]
            # sigma_tof = resolution / p.bw / p.tof_search_step / 3
            # sigma_aoa = resolution * (2 / p.rx_antennas) / p.aoa_cos_search_step / 3
            # ideal_output = scipy.ndimage.gaussian_filter(ideal_sparse.todense(),
            #                                              (sigma_tof, sigma_aoa)) * 2 * np.pi * sigma_aoa * sigma_tof
            # axs[1].imshow(ideal_output, aspect='auto', cmap=pl.get_cmap('jet'))
            # axs[1].set_title('ml ideal')
            # pl.show()
            # ml_peak = get_ranked_peaks(predicted_y[0].numpy(), p)
            # print('ml_peak aoa (deg)', np.rad2deg(ml_peak[:, 0]), 'tof (ns)', ml_peak[:, 1] * 1e9)
        if 'fft' in methods:
            # hmap = get_2d_heatmap(H[:, :, 0], p.fftsize, p.rx_antennas, p.bw, p.tof_search_step, p.tof_max,
            #                       p.aoa_cos_search_step)
            # pl.figure()
            # pl.imshow(hmap, aspect='auto', cmap=pl.get_cmap('jet'))
            # pl.title('fft')
            # pl.show()
            # fft_peak = get_ranked_peaks(hmap, p)
            # print('fft_peak aoa (deg)', np.rad2deg(fft_peak[:, 0]), 'tof (ns)', fft_peak[:, 1] * 1e9)
            hmap = r.get_aoa_tof_heatmap(H[:,:,0])
            peaks = r.filter_peaks(r.get_ranked_peaks(hmap, debug=True))
            r.show_aoa_tof_heatmap(hmap, peaks)
            # exit()


        sig_t2 = m2.get_new_channel_t(test1_par, target_snr_db=None)
        sig_f2 = np.fft.fft(sig_t2, axis=0)
        H2 = sig_f2 * m2.ltf_f[:, np.newaxis, np.newaxis]

        if 'mdtrack' in methods:
            v_list, runtime = run_mdtrack(m, sig_t[np.newaxis, :, :, :])
            sim_result['runtime_mdtrack'].append(runtime)
            print('mdtrack', m.print_pretty(v_list))
            print(np.abs([v.val for v in v_list[:, 4]]))
            plot_channel(m, sig_t, v_list)

            predicted_H = np.fft.fft(m2.get_new_channel_t(v_list)[:, :, 0], axis=0) * m.ltf_f[:, np.newaxis]
            beam_mdtrack = beamforming_gain_db(H2[:, :, 0], predicted_H)
        if 'ml' in methods:
        # ml_peak = np.array([[1.04719755e+00, 1.30000000e-07], [1.52077547e+00, 1.40000000e-07]])
            res, runtime = run_optml(r, p, H, model_2d)
            v_list = r.get_physical_path_parameters(res)
            sim_result['runtime_optml'].append(runtime)
            print('ml', m.print_pretty(v_list))

            plot_channel(m, sig_t, v_list)

            predicted_H = np.fft.fft(m2.get_new_channel_t(v_list)[:, :, 0], axis=0) * m.ltf_f[:, np.newaxis]
            beam_ml = beamforming_gain_db(H2[:, :, 0], predicted_H)

            sig_t -= m.get_new_channel_t(v_list)
        if 'fft' in methods:
            res, runtime = run_r2f2(r, p, H)
            v_list = r.get_physical_path_parameters(res)
            print('fft', m.print_pretty(v_list))
            plot_channel(m, sig_t, v_list)


        # beam_oracle = beamforming_gain_db(H2[:, :, 0], H2[:, :, 0])
        # print(f"beamforming gain (dB): \n"
        #     f"no beam: {beamforming_gain_db(H2[:, :, 0], np.ones((64, rx_ant))):.2f}"
        #     f"beam_mdtrack: {beam_mdtrack:.2f}, beam_ml: {beam_ml:.2f}, "
        #     f"oracle: {beam_oracle:.2f}")
        #
        # sim_result['bg_optml'].append(beam_ml)
        # sim_result['bg_mdtrack'].append(beam_mdtrack)
        # sim_result['bg_oracle'].append(beam_oracle)
        # p.pickle_save(sim_result, './sim_result.pckl')

        # pl.figure()
        # data1, data2 = np.sort(runtime_mdtrack), np.sort(runtime_optml)
        # pl.plot(data1, np.arange(0,1,1/len(runtime_mdtrack)))
        # pl.plot(data2, np.arange(0, 1, 1 / len(runtime_optml)))
        # pl.legend(['mdtrack', 'optml'])
        # pl.show()
        # print('mdtrack median: ', np.percentile(data1, 50), 'optml median: ', np.percentile(data2, 50))

def test_sim_large_fft():
    def interpolate(m, sig_f):
        a,b,c,d = sig_f.shape
        assert m.preamble_interval_sample % len(m.ltf_f) == 0
        ltf_per_interval = int(m.preamble_interval_sample/len(m.ltf_f))
        print(a, b, c, d, ltf_per_interval)
        res = np.zeros((a*ltf_per_interval, b, c, d), dtype=np.complex)
        for i in range(sig_f.shape[0]):
            if i == sig_f.shape[0] - 1:
                delta = (sig_f[i, :, :, :] - sig_f[i-1, :, :, :]) / ltf_per_interval
            else:
                delta = (sig_f[i+1,:,:,:] - sig_f[i,:,:,:]) / ltf_per_interval
            for j in range(ltf_per_interval):
                res[i*ltf_per_interval+j, :, :, :] = sig_f[i,:,:,:] + delta*j
        return res

    @timer
    def run_optml(r, p, H, model_2d):
        r.channel_est = H
        x = np.array([data_vectorization(data_normalization(H))])
        predicted_y = model_2d(x)
        hmap = predicted_y[0].numpy()
        peaks = p.filter_peaks(p.get_ranked_peaks(hmap, debug=True))
        p.show_aoa_tof_heatmap(hmap, peaks)
        res = r.solve_optimization(peaks.ravel())
        return r.get_physical_path_parameters(res)

    @timer
    def run_r2f2(r, p, H):
        r.channel_est = H
        hmap = r.get_aoa_tof_heatmap(H[:, :, 0])
        peaks = r.filter_peaks(r.get_ranked_peaks(hmap, debug=True))
        r.show_aoa_tof_heatmap(hmap, peaks)
        res = r.solve_optimization(peaks.ravel())
        return r.get_physical_path_parameters(res)

    def plot_channel(m, sig_t_orig, v_list):
        sig_t_est = m.get_new_channel_t(v_list)
        plt, axs = pl.subplots(2, m.rx_antennas)
        axs = axs.reshape(2, m.rx_antennas)
        for i in range(m.rx_antennas):
            ch1 = np.fft.fft(sig_t_orig[:, i, 0]) * m.ltf_f
            ch2 = np.fft.fft(sig_t_est[:, i, 0]) * m.ltf_f

            ch1 = np.fft.fftshift(ch1)
            ch1 = ch1[np.abs(ch1) > 1e-12]
            ch2 = np.fft.fftshift(ch2)
            ch2 = ch2[np.abs(ch2) > 1e-12]

            axs[0, i].plot(10 * np.log10(np.abs(ch1)))
            axs[0, i].plot(10 * np.log10(np.abs(ch2)))
            axs[0, i].set_xlabel('subcarriers')
            axs[0, i].set_ylabel('Magnitude (dB)')
            axs[0, i].set_title(f"antenna {i + 1}")
            axs[1, i].plot(np.angle(ch1))
            axs[1, i].plot(np.angle(ch2))
            axs[1, i].set_xlabel('subcarriers')
            axs[1, i].set_ylabel('Phase (radian)')
            axs[1, i].set_title(f"antenna {i + 1}")

            if i == 0:
                axs[0, i].legend(['measured', 'fitted'])
                # axs[1, i].legend(['measured', 'fitted'])
        pl.show()

    preamble_repeat_cnt = 40
    rx_ant = 4
    amin, L, H = 0.1, 3, 1
    resolution = 1.0
    fc = 5310e6
    bw = 40e6
    md_config = {'rx_antennas': rx_ant, 'tx_antennas': 1, 'tof_max': 64/bw, 'aod_max': 0.02,
                 'initial_est_stop_threshold_db': 1.0, 'debug': True,
                 'fc': fc, 'bw': bw,
                 'dop_max': preamble_repeat_cnt / 2, 'dop_search_step': 1,
                 'preamble_interval_sec': 1/preamble_repeat_cnt,
                 'preamble_repeat_cnt': preamble_repeat_cnt}
    m = Mdtrack(**md_config)
    config = {'rx_antennas': rx_ant, 'tx_antennas': 1, 'fc': fc, 'bw': bw,
              'tof_max': 250e-9, 'tof_search_step': 5e-9, 'aoa_cos_search_step': 0.05,
              'nn_sparse_y': True, 'nn_data_snr_range': (15, 25), 'nn_data_size': 100000, 'nn_epochs': 1000,
              'max_physical_paths': 5}
    p = Optml2(**config)

    config = {'rx_antennas': rx_ant, 'tx_antennas': 1, 'fc': fc, 'bw': bw,
              'tof_max': 250e-9, 'tof_search_step': 0.5e-9,
              'aoa_search_step': 0.01, 'dop_max': 0.1, 'debug': False}
    r = R2f2(**config)

    model_2d = keras.models.load_model(
        f"./optml2/test_2d_aoa_tof_resolution/{rx_ant}rx/res{resolution:.2f}/L{L}H{H}_amin{amin}/model.h5")

    test1_par = np.array([[np.deg2rad(60.7), np.deg2rad(0), 17.6e-9, 0.0, 0.8],
                          [np.deg2rad(130), np.deg2rad(0), 45.1e-9, 0.0, 0.8 / 5],
                          [np.deg2rad(117), np.deg2rad(0), 65.1e-9, 15.0, 0.8 / 8],
                          [np.deg2rad(80), np.deg2rad(0), 85.1e-9, 9.0, 0.8 / 10]])
    sig_t = m.get_new_preamble_train_t(test1_par, target_snr_db=25)
    # m.resolve_multipath(sig_t)
    # exit()
    sig_f = np.fft.fft(sig_t, axis=1)
    interpolated_sig_f = interpolate(m, sig_f)
    a,b,c,d = interpolated_sig_f.shape
    interpolated_sig_t = np.fft.ifft(interpolated_sig_f, axis=1).reshape(a*b,c,d)

    # large fft
    new_sig_f = np.fft.fft(interpolated_sig_t, axis=0)

    pl.figure()
    pl.plot(np.abs(new_sig_f[int(40e6/64)-50:int(40e6/64)+50,0,0]))
    pl.show()

    reconstructed_sig_t = np.zeros((b, c, d), dtype=np.complex)
    idx = 40e6/64 * np.arange(64)
    nondop_lts = np.fft.ifft(new_sig_f[idx.astype(int), :, :], axis=0)
    nondop_power = np.array([np.linalg.norm(nondop_lts[:, i, 0]) for i in range(c)])
    dop_detected = []
    for dop in m.dop_search_range:
        if dop != 0:
            tmp = np.fft.ifft(new_sig_f[(int(dop) + idx.astype(int)) % new_sig_f.shape[0], :, :], axis=0)
            tmp_power = np.array([np.linalg.norm(tmp[:, i, 0]) for i in range(c)])
            print(f"dop={dop}, power={np.round(tmp_power / nondop_power, 5)}")
            if np.all(tmp_power / nondop_power > 1 / 20):
                reconstructed_sig_t += tmp
                dop_detected.append(dop)
    print(f"dop_detected: {dop_detected}")

    reconstructed_sig_f = np.fft.fft(reconstructed_sig_t, axis=0)
    H_mobile = reconstructed_sig_f * m.ltf_f[:, np.newaxis, np.newaxis]
    H_all = sig_f[0,:,:,:] * m.ltf_f[:, np.newaxis, np.newaxis]

    hmap = p.get_aoa_tof_heatmap(H_all[:, :, 0])
    hmap2 = p.get_aoa_tof_heatmap(H_mobile[:, :, 0])
    x = np.array([data_vectorization(data_normalization(H_all))])
    hmap3 = model_2d(x)[0]
    x = np.array([data_vectorization(data_normalization(H_mobile))])
    hmap4 = model_2d(x)[0]

    plt, axs = pl.subplots(2,2)
    axs[0,0].imshow(hmap, aspect='auto', cmap=pl.get_cmap('jet'))
    axs[0,0].set_title('fft all paths')
    axs[1,0].imshow(hmap2, aspect='auto', cmap=pl.get_cmap('jet'))
    axs[1,0].set_title('fft mobile paths')
    axs[0, 1].imshow(hmap3, aspect='auto', cmap=pl.get_cmap('jet'))
    axs[0, 1].set_title('ml all paths')
    axs[1, 1].imshow(hmap4, aspect='auto', cmap=pl.get_cmap('jet'))
    axs[1, 1].set_title('ml mobile paths')
    pl.show()

    v_list, runtime = run_r2f2(r, p, H_mobile)
    print('r2f2', m.print_pretty(v_list))
    plot_channel(m, reconstructed_sig_t, v_list)

    v_list, runtime = run_optml(r, p, H_all, model_2d)
    print('ml 1st', m.print_pretty(v_list))
    plot_channel(m, sig_t[0,:,:,:], v_list)
    t2 = m.get_new_channel_t(v_list, target_snr_db=None)
    f2 = t2 * m.ltf_f[:, np.newaxis, np.newaxis]
    v_list, runtime = run_optml(r, p, H_all - f2, model_2d)
    print('ml 2nd', m.print_pretty(v_list))
    # plot_channel(m, sig_t[0, :, :, :], v_list)

    # v_list, runtime = run_optml(r, p, H_mobile, model_2d)
    # print('ml', m.print_pretty(v_list))
    # plot_channel(m, reconstructed_sig_t, v_list)
    # v_list, runtime = m.resolve_multipath(sig_t)
    # print('mdtrack', m.print_pretty(v_list))
    # plot_channel(m, reconstructed_sig_t, v_list)
    exit()

def test_experiment_large_fft():
    def interpolate(m, sig_f):
        a,b,c,d = sig_f.shape
        assert m.preamble_interval_sample % len(m.ltf_f) == 0
        ltf_per_interval = int(m.preamble_interval_sample/len(m.ltf_f))
        print(a, b, c, d, ltf_per_interval)
        res = np.zeros((a*ltf_per_interval, b, c, d), dtype=np.complex)
        for i in range(sig_f.shape[0]):
            if i == sig_f.shape[0] - 1:
                delta = (sig_f[i, :, :, :] - sig_f[i-1, :, :, :]) / ltf_per_interval
            else:
                delta = (sig_f[i+1,:,:,:] - sig_f[i,:,:,:]) / ltf_per_interval
            for j in range(ltf_per_interval):
                res[i*ltf_per_interval+j, :, :, :] = sig_f[i,:,:,:] + delta*j
        return res

    def load_lts_train(m, mainfolder, subfolder, tof_shift_second, start_index):
        preamble_train = []
        rf_calibration = scipy.io.loadmat(f"{mainfolder}/{subfolder}/rf_calibration.mat")['rf_calibration']
        for i in range(start_index, start_index + m.preamble_repeat_cnt):
            lts_samples = scipy.io.loadmat(f"{mainfolder}/{subfolder}/lts_samples_{i}.mat")['lts_samples']
            H = np.fft.fft(lts_samples, axis=0) * rf_calibration
            H = H * m.get_tof_effect(tof_shift_second)[:, np.newaxis]
            H_t = np.fft.ifft(H, axis=0)
            # LOS = m.get_new_channel_t([[np.deg2rad(137.51), 0, 85.5e-9, 0, 4.59*np.exp(1j*-0.23)]])
            # H_t = H_t - LOS[:,:,0]
            preamble_train.append(H_t[:, :, np.newaxis])

        return np.array(preamble_train)

    def load_lts_train2(m, mainfolder, subfolder, tof_shift_second, preamble_cnt):
        preamble_train = []
        rf_calibration = mat73.loadmat(f"{mainfolder}/{subfolder}/rf_calibration.mat")['rf_calibration']
        lts_samples = mat73.loadmat(f"{mainfolder}/{subfolder}/lts_sample_train.mat")['lts_sample_train'].T
        assert m.bw % (64*preamble_cnt) == 0
        interval = int(m.bw / 64 / preamble_cnt)
        for i in range(0, int(m.bw/64), interval):
            H = np.fft.fft(lts_samples[i*64:(i+1)*64, :], axis=0) * rf_calibration
            H = H * m.get_tof_effect(tof_shift_second)[:, np.newaxis]
            H_t = np.fft.ifft(H, axis=0)
            preamble_train.append(H_t[:, :, np.newaxis])

        return np.array(preamble_train)

    @timer
    def run_optml(r, p, H, model_2d):
        r.channel_est = H
        x = np.array([data_vectorization(data_normalization(H))])
        predicted_y = model_2d(x)
        hmap = predicted_y[0].numpy()
        peaks = p.filter_peaks(p.get_ranked_peaks(hmap, debug=True))
        p.show_aoa_tof_heatmap(hmap, peaks)
        res = r.solve_optimization(peaks.ravel())
        return r.get_physical_path_parameters(res)

    @timer
    def run_r2f2(r, p, H):
        r.channel_est = H
        hmap = r.get_aoa_tof_heatmap(H[:, :, 0])
        peaks = r.filter_peaks(r.get_ranked_peaks(hmap, debug=True))
        r.show_aoa_tof_heatmap(hmap, peaks)
        res = r.solve_optimization(peaks.ravel())
        return r.get_physical_path_parameters(res)

    def plot_channel(m, sig_t_orig, v_list):
        sig_t_est = m.get_new_channel_t(v_list)
        plt, axs = pl.subplots(2, m.rx_antennas)
        axs = axs.reshape(2, m.rx_antennas)
        for i in range(m.rx_antennas):
            ch1 = np.fft.fft(sig_t_orig[:, i, 0]) * m.ltf_f
            ch2 = np.fft.fft(sig_t_est[:, i, 0]) * m.ltf_f

            ch1 = np.fft.fftshift(ch1)
            ch1 = ch1[np.abs(ch1) > 1e-12]
            ch2 = np.fft.fftshift(ch2)
            ch2 = ch2[np.abs(ch2) > 1e-12]

            axs[0, i].plot(10 * np.log10(np.abs(ch1)))
            axs[0, i].plot(10 * np.log10(np.abs(ch2)))
            axs[0, i].set_xlabel('subcarriers')
            axs[0, i].set_ylabel('Magnitude (dB)')
            axs[0, i].set_title(f"antenna {i + 1}")
            axs[1, i].plot(np.angle(ch1))
            axs[1, i].plot(np.angle(ch2))
            axs[1, i].set_xlabel('subcarriers')
            axs[1, i].set_ylabel('Phase (radian)')
            axs[1, i].set_title(f"antenna {i + 1}")

            if i == 0:
                axs[0, i].legend(['measured', 'fitted'])
                # axs[1, i].legend(['measured', 'fitted'])
        pl.show()

    preamble_repeat_cnt = 40
    rx_ant = 3
    amin, L, H = 0.2, 3, 1
    resolution = 1.0
    fc = 5310e6
    bw = 40e6
    # mainfolder = './matlab_scripts/experiment_data'
    mainfolder = 'C:/Users/alex/Desktop/WARPLab_Reference_Design_7.7.1/M_Code_Examples/matlab_scripts/experiment_data'
    # subfolder = 'lts_sample_train_walk_close_5310e6_40e6'
    subfolder = ''
    noise_pwr = scipy.io.loadmat(f"{mainfolder}/{subfolder}/adjusted_noisepwr_{1}.mat")['adjusted_noisepwr'][0]
    # noise_pwr = mat73.loadmat(f"{mainfolder}/{subfolder}/noise_pwr.mat")['noise_pwr']
    md_config = {'rx_antennas': rx_ant, 'tx_antennas': 1, 'tof_max': 64/bw, 'aod_max': 0.02,
                 'initial_est_stop_threshold_db': 1.0, 'debug': True,
                 'fc': fc, 'bw': bw, 'rf_chain_noise_pwr': noise_pwr,
                 'dop_max': preamble_repeat_cnt / 2, 'dop_search_step': 1,
                 'preamble_interval_sec': 1/preamble_repeat_cnt,
                 'preamble_repeat_cnt': preamble_repeat_cnt}
    m = Mdtrack(**md_config)
    config = {'rx_antennas': rx_ant, 'tx_antennas': 1, 'fc': fc, 'bw': bw,
              'tof_max': 250e-9, 'tof_search_step': 5e-9, 'aoa_cos_search_step': 0.05,
              'nn_sparse_y': True, 'nn_data_snr_range': (15, 25), 'nn_data_size': 100000, 'nn_epochs': 1000,
              'max_physical_paths': 5}
    p = Optml2(**config)

    config = {'rx_antennas': rx_ant, 'tx_antennas': 1, 'fc': fc, 'bw': bw,
              'tof_max': 250e-9, 'tof_search_step': 0.5e-9,
              'aoa_search_step': 0.01, 'dop_max': 0.1, 'debug': True}
    r = R2f2(**config)


    # model_2d = keras.models.load_model(
    #     f"./optml2/test_2d_aoa_tof_resolution/{rx_ant}rx/res{resolution:.2f}/L{L}H{H}_amin{amin}/model.h5")

    # load data
    do_interpolation = True
    reload_from_disk = True
    tof_shift_second = 2 / bw

    if do_interpolation:
        if reload_from_disk:
            start_index = 1
            preamble_train = load_lts_train(m, mainfolder, subfolder, tof_shift_second, start_index)

            # preamble_train = load_lts_train2(m, mainfolder, subfolder, tof_shift_second,
            #                                  preamble_cnt=m.preamble_repeat_cnt)
            # scipy.io.savemat(f"{mainfolder}/{subfolder}/processed_lts_sample_train.mat",
            #                  {'processed_lts_sample_train': preamble_train})
        else:
            preamble_train = scipy.io.loadmat(f"{mainfolder}/{subfolder}/processed_lts_sample_train.mat")[
                'processed_lts_sample_train']
        sig_f = np.fft.fft(preamble_train, axis=1)
        interpolated_sig_f = interpolate(m, sig_f)
        a, b, c, d = interpolated_sig_f.shape
        interpolated_sig_t = np.fft.ifft(interpolated_sig_f, axis=1).reshape(a * b, c, d)
    else:
        preamble_train = load_lts_train2(m, mainfolder, subfolder, tof_shift_second, preamble_cnt=int(m.bw/64))
        a, b, c, d = preamble_train.shape
        interpolated_sig_t = preamble_train.reshape(a*b,c,d)

    # large fft
    print("large fft input shape: ", interpolated_sig_t.shape)
    sig_f = np.fft.fft(interpolated_sig_t, axis=0)

    plt, axs = pl.subplots(1, c)
    for i in range(c):
        axs[i].plot(range(-50,50), np.abs(sig_f[int(bw/64)-50:int(bw/64)+50, i]))
        axs[i].set_title(f"RF {i+1}")
    pl.show()
    # exit()

    reconstructed_sig_t = np.zeros((b, c, d), dtype=np.complex)
    idx = 40e6/64 * np.arange(64)
    nondop_lts = np.fft.ifft(sig_f[idx.astype(int), :, :], axis=0)
    nondop_power = np.array([np.linalg.norm(nondop_lts[:,i,0]) for i in range(c)])
    dop_detected = []
    for dop in m.dop_search_range:
        if dop !=0:
            tmp = np.fft.ifft(sig_f[(int(dop) + idx.astype(int)) % sig_f.shape[0], :, :], axis=0)
            tmp_power = np.array([np.linalg.norm(tmp[:,i,0]) for i in range(c)])
            print(f"dop={dop}, power={np.round(tmp_power/nondop_power, 5)}")
            if np.all(tmp_power/nondop_power > 1/20):
                reconstructed_sig_t += tmp
                dop_detected.append(dop)
    print(f"dop_detected: {dop_detected}")
    reconstructed_sig_f = np.fft.fft(reconstructed_sig_t, axis=0)
    H_mobile = reconstructed_sig_f * m.ltf_f[:, np.newaxis, np.newaxis]
    H_all = np.fft.fft(preamble_train[0,:,:,:], axis=0) * m.ltf_f[:, np.newaxis, np.newaxis]

    # hmap = p.get_aoa_tof_heatmap(H_all[:,:,0])
    # hmap2 = p.get_aoa_tof_heatmap(H_mobile[:,:,0])
    # x = np.array([data_vectorization(data_normalization(H_all))])
    # hmap3 = model_2d(x)[0]
    # x = np.array([data_vectorization(data_normalization(H_mobile))])
    # hmap4 = model_2d(x)[0]
    #
    # plt, axs = pl.subplots(2,2)
    # axs[0,0].imshow(hmap, aspect='auto', cmap=pl.get_cmap('jet'))
    # axs[0,0].set_title('fft all paths')
    # axs[1,0].imshow(hmap2, aspect='auto', cmap=pl.get_cmap('jet'))
    # axs[1,0].set_title('fft mobile paths')
    # axs[0, 1].imshow(hmap3, aspect='auto', cmap=pl.get_cmap('jet'))
    # axs[0, 1].set_title('ml all paths')
    # axs[1, 1].imshow(hmap4, aspect='auto', cmap=pl.get_cmap('jet'))
    # axs[1, 1].set_title('ml mobile paths')
    # pl.show()

    v_list = r.resolve_multipath(preamble_train[0,:,:,:])
    print('r2f2 all path', m.print_pretty(v_list))
    plot_channel(m, preamble_train[0,:,:,:], v_list)

    if dop_detected:
        v_list = r.resolve_multipath(reconstructed_sig_t)
        print('r2f2 mobile path ', m.print_pretty(v_list))
        plot_channel(m, reconstructed_sig_t, v_list)

    v_list = m.resolve_multipath(preamble_train)
    print('mdtrack all path', m.print_pretty(v_list))
    plot_channel(m, preamble_train[0,:,:,:], v_list)

    # v_list, runtime = run_optml(r, p, H_mobile, model_2d)
    # print('ml', m.print_pretty(v_list))
    # plot_channel(m, reconstructed_sig_t, v_list)
    exit()

if __name__ == '__main__':
    # train_1d_tof_resolution()
    # check_1d_tof_resolution()
    # train_2d_aoa_tof_resolution()
    # check_2d_aoa_tof_resolution()
    test_on_experiment_data()
    # test_on_sim_data()
    # test_sim_large_fft()
    # test_experiment_large_fft()