from mdtrack_mobicom19.core import Mdtrack
from utils.common import timer, Snr, Db
from utils.siggen import Signal
from r2f2_sigcomm16.core import R2f2
import numpy as np
import matplotlib.pyplot as pl
import logging
from scipy.special import diric
from scipy import signal
import seaborn as sns

logger = logging.getLogger(__name__)


def test_correctness_and_timer():
    @timer
    def run_mdtrack(m, sig_t):
        return m.resolve_multipath(sig_t)

    config = {'rx_antennas': 4, 'tx_antennas': 1, 'tof_max': 300e-9, 'dop_max': 20, 'preamble_interval_sec': 25e-3,
              'initial_est_stop_threshold_db': 1.0, 'preamble_repeat_cnt': 40,
              'debug': True}

    m = Mdtrack(**config)
    test1_par = np.array([[np.deg2rad(60.7), np.deg2rad(40), 20.6e-9, 0.0, 1],
                          [np.deg2rad(75), np.deg2rad(50), 32.1e-9, 0.0, 0*0.2],
                          [np.deg2rad(73.8), np.deg2rad(50), 28.1e-9, 15.0, 0.1]])
    # sig_t = m.get_new_channel_t(test1_par, target_snr_db=20)
    sig_t = m.get_new_preamble_train_t(test1_par, target_snr_db=20)
    m.show_freq_response(sig_t[0])
    v_list, runtime = run_mdtrack(m, sig_t)
    logger.info(m.print_pretty(v_list))
    return


def test_detect_doppler_shift():
    config = {'rx_antennas': 1, 'tx_antennas': 1, 'aod_max': 0.05, 'dop_max': 0.2, 'debug': True}
    m = Mdtrack(**config)
    # test2_par = np.array([[60.7 / 180 * np.pi, 0.0, 20.6e-9, 0.0, 0.6]])
    sig_f = m.get_tof_effect(20.6e-9)
    sig_t = np.fft.ifft(sig_f)
    sig_t = np.tile(sig_t, 4)
    p = np.sum(np.abs(sig_t) ** 2)

    # apply doppler shift
    ds = lambda x: np.exp(1j * 2 * np.pi * x / m.bw * np.arange(len(sig_t)))
    sig_t *= ds(10)

    # correlation
    cor = [np.abs(np.vdot(ds(k), sig_t)) / p for k in range(150)]
    # cfo_est = np.mean(np.angle(sig_t[:len(sig_f)].conjugate()*sig_t[len(sig_f):])) / (2*np.pi*len(sig_f)/m.bw)
    # print(f"{cfo_est}")
    print(np.argmax(cor))

    # sig_f = np.fft.fft(sig_t)
    # plt, axs = pl.subplots(2, 2, figsize=(10, 10))
    # axs[0,0].plot(np.abs(sig_f))
    # axs[0,1].stem(np.angle(sig_f))
    # # axs[1,0].plot(np.abs(sig_t))
    # axs[1,1].stem(cor)
    # pl.show()


def test_detect_doppler_shift_spaced_preambles():
    preamble_interval_sec = 20e-3
    preamble_repeat_cnt = 50
    test_doppler_shift_val = 8  # Hz
    config = {'rx_antennas': 4, 'tx_antennas': 1, 'tof_max': 64*25e-9, 'aod_max': 0.05,
              'dop_max': 1 / preamble_interval_sec / 2, 'dop_search_step': 1,
              'preamble_interval_sec': preamble_interval_sec,
              'preamble_repeat_cnt': preamble_repeat_cnt,
              'debug': True, 'fc': 5310e6, 'bw': 40e6, 'initial_est_stop_threshold_db': 0.0}
    m = Mdtrack(**config)

    # test1_par = np.array([[np.deg2rad(0.0), np.deg2rad(0), 4e-9, 0.0, 1],
    #                       [np.deg2rad(0.0), np.deg2rad(0), 4e-9 + (m.wavelength[63]+m.wavelength[62])/4/3e8, 0.0, 1]])
    test1_par = np.array([[np.deg2rad(84.7), np.deg2rad(0), 20e-9, 0.0, 1],
                          [np.deg2rad(103.8), np.deg2rad(0), 20.1e-9, 20.0, 0.1]])
    # H_t = m.get_new_channel_t(test1_par, target_snr_db=15)
    # m.show_freq_response(H_t)
    # m.initial_estimation(H_t)
    # m.gen_heatmap(H_t, v_list=test1_par, plot=True, save='')
    # exit()
    # m.resolve_multipath(H_t[np.newaxis, :, :, :])
    # print(m)
    # exit()

    sig_t = m.get_new_preamble_train_t(test1_par, target_snr_db=None)
    # v_list = m.resolve_multipath(sig_t)
    # exit()

    # plot measured and fitted
    # v_list = m.resolve_multipath(H_t[np.newaxis, :, :, :])
    # sig_t_est = m.get_new_channel_t(v_list)
    # plt, axs = pl.subplots(2, m.rx_antennas)
    # axs = axs.reshape(2, m.rx_antennas)
    # for i in range(m.rx_antennas):
    #     ch1 = np.fft.fft(H_t[:, i, 0]) * m.ltf_f
    #     ch2 = np.fft.fft(sig_t_est[:, i, 0]) * m.ltf_f
    #
    #     ch1 = np.fft.fftshift(ch1)
    #     ch1 = ch1[np.abs(ch1) > 1e-12]
    #     ch2 = np.fft.fftshift(ch2)
    #     ch2 = ch2[np.abs(ch2) > 1e-12]
    #
    #     axs[0, i].plot(10 * np.log10(np.abs(ch1)))
    #     axs[0, i].plot(10 * np.log10(np.abs(ch2)))
    #     axs[0, i].set_xlabel('subcarriers')
    #     axs[0, i].set_ylabel('Magnitude (dB)')
    #     axs[0, i].set_title(f"antenna {i + 1}")
    #     axs[1, i].plot(np.angle(ch1))
    #     axs[1, i].plot(np.angle(ch2))
    #     axs[1, i].set_xlabel('subcarriers')
    #     axs[1, i].set_ylabel('Phase (radian)')
    #     axs[1, i].set_title(f"antenna {i + 1}")
    #
    #     if i == 0:
    #         axs[0, i].legend(['measured', 'fitted'])
    #         # axs[1, i].legend(['measured', 'fitted'])
    # pl.show()
    # exit()

    # doppler shift of each antenna
    # sig_t = np.fft.ifft(np.fft.fft(sig_t, axis=1) * m.ltf_f[np.newaxis, :, np.newaxis, np.newaxis], axis=1)
    plt, axs = pl.subplots(int(np.ceil(m.rx_antennas/2)), 2)
    axs = axs.reshape(int(np.ceil(m.rx_antennas/2)), 2)
    for i in range(m.rx_antennas):
        cor = [np.abs(np.vdot(x, sig_t[:, :, i, 0])) for x in m.dop_search_mat]
        axs[i//2, i%2].stem(cor)
        print(f"antenna {i+1}, {m.dop_search_range[20]}/{m.dop_search_range[29]}: {10*np.log10(cor[20]/cor[29])}")
        # axs[i].stem(cor)
        xtick_pos = np.arange(0, len(m.dop_search_range), len(m.dop_search_range) // 5)
        axs[i//2, i%2].set_xticks(xtick_pos)
        axs[i//2, i%2].set_xticklabels([str(x) for x in m.dop_search_range[xtick_pos]])
        axs[i//2, i%2].set_title(f"rx_antenna {i + 1}")
        axs[i//2, i%2].set_xlabel(f"doppler shift (Hz)")
        axs[i // 2, i % 2].set_ylabel(f"abs")
    pl.show()
    exit()

    # v_list_train = m.resolve_multipath_preamble_train(sig_t)
    # aoa, dop = [], []
    # for v_list in v_list_train:
    #     for v in v_list:
    #         aoa.append(np.rad2deg(v[0].val))
    #         dop.append(v[3].val)
    # aoa, dop = np.array(aoa), np.array(dop)
    # sns.jointplot(data={'aoa': aoa, 'doppler': dop}, x='aoa', y='doppler', xlim=(0, 180), ylim=(-m.dop_max, m.dop_max))
    # fig, axs = pl.subplots(2, 1)
    # axs[0].hist(aoa)
    # axs[0].set_xlabel('aoa (deg)')
    # axs[1].hist(dop)
    # axs[1].set_xlabel('doppler (Hz)')
    # pl.show()
    # exit()

    sig_t = sig_t[:, :, 0, 0]
    ch_est = np.fft.fft(sig_t, axis=1) * m.ltf_f[np.newaxis, :]
    sig_t = np.fft.ifft(ch_est, axis=1)
    cor = [np.abs(np.vdot(x, sig_t)) for x in m.dop_search_mat]
    print(np.argmax(cor))

    pl.figure()
    pl.stem(cor)
    xtick_pos = np.arange(0, len(m.dop_search_range), len(m.dop_search_range) // 10)
    pl.xticks(xtick_pos, [str(x) for x in m.dop_search_range[xtick_pos]])
    pl.xlabel('Doppler shift (Hz)')
    pl.show()
    exit()

    # generate test sig, the preamble train
    # test1_par = np.array([[np.deg2rad(60.7), np.deg2rad(40), 20.6e-9, 0.0, 1],
    #                       [np.deg2rad(73.8), np.deg2rad(50), 28.1e-9, 0.0, 0.0]])
    # sig_t = m.get_new_channel_t(test1_par, target_snr_db=20)
    # # m.show_freq_response(sig_t)
    # sig_t = sig_t[:, 0, 0]

    sig_f = m.get_tof_effect(37.6e-9) * np.abs(m.ltf_f)
    sig_t = np.fft.ifft(sig_f)
    # sig_t = np.tile(sig_t, 2)
    preamble = np.array([sig_t for _ in range(preamble_repeat_cnt)])
    preamble_pwr = np.sum(np.abs(preamble) ** 2)
    preamble_interval_sample = int(preamble_interval_sec * m.bw)  # 25ms equals 500e3 samples at 20 Mhz sampling rate
    print('fmax', m.bw / 2 / preamble_interval_sample)

    # helper for applying doppler shift to the preamble train
    def doppler(ds, preamble, sig_t, preamble_interval_sample):
        dop = []
        for i in range(len(preamble)):
            idx_start = i * preamble_interval_sample
            dop.append(np.exp(1j * 2 * np.pi * ds / m.bw * np.arange(idx_start, idx_start + len(sig_t))))
        return preamble * np.array(dop)

    test_sig_t = doppler(test_doppler_shift_val, preamble, sig_t, preamble_interval_sample)

    # correlation
    cor = [np.abs(np.vdot(x, test_sig_t)) for x in m.dop_search_mat]
    print(np.argmax(cor))

    pl.figure()
    pl.stem(cor)
    xtick_pos = np.arange(0, len(m.dop_search_range), len(m.dop_search_range) // 10)
    pl.xticks(xtick_pos, [str(x) for x in m.dop_search_range[xtick_pos]])
    pl.xlabel('Doppler shift (Hz)')
    pl.show()
    # cfo_est = np.mean(np.angle(sig_t[:len(sig_f)].conjugate()*sig_t[len(sig_f):])) / (2*np.pi*len(sig_f)/m.bw)
    # print(f"{cfo_est}")


def test_tof_correlation():
    config = {'rx_antennas': 1, 'tx_antennas': 1,
              'tof_max': 100e-9, 'aod_max': 0.05, 'dop_max': 0.2, 'debug': True,
              'initial_est_stop_threshold_db': 1.0}
    m = Mdtrack(**config)
    test_par = np.array([[np.deg2rad(45), 0.0, 78.8e-9, 0.0, 0.4 * np.exp(1j * 2)],
                         [np.deg2rad(60), 0.0, 111.6e-9, 0.0, 0.5 * np.exp(1j * 2)],
                         [np.deg2rad(60), 0.0, 162.6e-9, 0.0, 0.9 * np.exp(1j * 2)]])
    # test_par1 = np.array([[np.deg2rad(45), 0.0, 50.8e-9, 0.0, 1.0 * np.exp(1j * 2)]])
    # test_par2 = np.array([[np.deg2rad(60), 0.0, 65.6e-9, 0.0, 0.1 * np.exp(1j * 2)]])
    sig_t = m.get_new_channel_t(test_par, target_snr_db=20)
    # sig_t1 = m.get_new_channel_t(test_par1, target_snr_db=20)
    # sig_t2 = m.get_new_channel_t(test_par2, target_snr_db=20)
    ch_est = np.fft.fft(sig_t, axis=0) * m.ltf_f[:, np.newaxis, np.newaxis]
    # ch_est1 = np.fft.fft(sig_t1, axis=0) * m.ltf_f[:, np.newaxis, np.newaxis]
    # ch_est2 = np.fft.fft(sig_t2, axis=0) * m.ltf_f[:, np.newaxis, np.newaxis]

    N = 256
    f0 = np.fft.ifft(ch_est[:, 0, 0], N)
    f02 = np.fft.ifft(ch_est[:, 0, 0] * signal.windows.gaussian(64, 64), N)
    # f1 = np.fft.ifft(ch_est1[:, 0, 0], N)
    # f2 = np.fft.ifft(ch_est2[:, 0, 0], N)
    # f3 = np.abs(np.fft.ifft(ch_est1[:, 0, 0] * signal.windows.hann(64), 1024))
    # f4 = np.abs(np.fft.ifft(ch_est2[:, 0, 0] * signal.windows.hann(64), 1024))
    plt, axs = pl.subplots(2, 1)
    db = lambda x: 20 * np.log10(np.abs(x))
    # pl.figure()
    axs[0].plot(np.abs(f0))
    # axs[0].plot(db(f2))
    # axs[0].plot(db(f1))
    axs[1].plot(np.abs(f02))
    # axs[1].plot(db(f3))
    # axs[1].plot(db(f4))
    # pl.stem(f2)
    pl.show()

    # # gen heatmaps
    # heatmaps = [m.gen_heatmap(sig_t)]
    # tmp_sig = sig_t.copy()
    # for i in range(len(initial)):
    #     tmp_sig -= m.get_reconstructed_sig(initial[i])
    #     heatmaps.append(m.gen_heatmap(tmp_sig))
    #
    # plt, axs = pl.subplots(1, len(heatmaps))
    # for i in range(len(heatmaps)):
    #     axs[i].imshow(np.abs(heatmaps[i]), aspect='auto', cmap=pl.get_cmap('jet'))
    # pl.show()

    exit()
    p = np.sum(np.abs(m.ltf_t) ** 2)
    # sig_t = np.fft.ifft(sig_f)
    # sig_t = np.tile(sig_t, 2)

    test_sig_t = sig_t[:, :, 0]
    # step = 0.5
    # tof_range = np.arange(0, 100, step)
    # correlation
    aoa_cor = [np.linalg.norm(np.sum(k.conjugate() * test_sig_t, axis=1)) ** 2 / p for k in m.aoa_search_mat]
    print(m.aoa_search_range[np.argmax(aoa_cor)] / np.pi * 180)

    # test_sig_t *= m.aoa_search_mat[np.argmax(aoa_cor)].conjugate()
    tof_cor = np.array(
        [np.sum(np.abs(k[:, np.newaxis].conjugate() * test_sig_t), axis=0) / p for k in m.tof_search_vec])
    print([m.tof_search_range[x] * 1e9 for x in np.argmax(tof_cor, axis=0)])
    # plt, axs = pl.subplots(2,3)
    # axs[0,0].stem(aoa_cor)
    # for i in range(config['rx_antennas']):
    #     axs[1, i].stem(tof_cor[:,i])
    # pl.show()


def test_aoa_correlation():
    np.set_printoptions(suppress=True)

    def dirichlet(w, N, center=0):
        return np.exp(1j * (w - center) * (N - 1) / 2) * diric(w - center, N)

    preamble_repeat_cnt = 1
    config = {'rx_antennas': 3, 'tx_antennas': 1, 'tof_max': 10 * 25e-9, 'aod_max': 0.02,
                 'initial_est_stop_threshold_db': 3.0, 'debug': True, 'fc': 5310e6, 'bw': 40e6,
                 'dop_max': preamble_repeat_cnt / 2, 'dop_search_step': 1,
                 'preamble_interval_sec': 1 / preamble_repeat_cnt,
                 'preamble_repeat_cnt': preamble_repeat_cnt}

    m = Mdtrack(**config)
    # r = R2f2(**config)
    # test_par = np.array([[40.0 / 180 * np.pi, 0.0, 20.6e-9, 0.0, 0.9 * np.exp(1j * 2)],
    #                      [67 / 180 * np.pi, 0.0, 52.6e-9, 0.0, 0.5 * np.exp(1j * 2)]])
    test_par = np.array([[np.deg2rad(40.0), 0.0, 20.6e-9, 0.0, 0.9 * np.exp(1j * 2)]])
    # sig_t = m.get_new_channel_t(test_par, target_snr_db=20)
    # ch_est = np.fft.fft(sig_t, axis=0) * m.ltf_f[:, np.newaxis, np.newaxis]
    # aoa_ifft = r.rx_F_inv[5,:,:] @ ch_est[5, :, 0]
    # aoa_ifft = np.fft.ifft(ch_est[5, :, 0])

    # pl.plot(np.arange(32), np.abs(np.fft.ifft(sig_f, n=32, axis=1).T))
    # print(m.wavelength/m.ant_spacing)
    # pl.show()
    # exit()

    # sig_t = np.fft.ifft(sig_f[5, :])
    # w = 2 * np.pi * np.arange(N) / N
    # center1 = 2 * np.pi / N * 0.66 / (2 / N)
    # center2 = 2 * np.pi / N * 0.81 / (2 / N)
    # diri = dirichlet(w, N, center1) + dirichlet(w, N, center2)
    # print(sig_t - diri)
    # plt, axs = pl.subplots(2, 1)
    # axs[0].stem(np.abs(sig_t), use_line_collection=True)
    # axs[1].stem(np.abs(diri), use_line_collection=True)
    # pl.show()
    # exit()

    # m.resolve_multipath(sig_t)
    # m.initial_estimation2(sig_t)
    # m.show_heatmap()

    cor = [np.linalg.norm(np.sum(x.conjugate()[:, :, np.newaxis] * ch_est, axis=1)) for x in m.aoa_search_mat]

    lamb = 3e8 / m.fc
    aperture = (m.rx_antennas) * lamb / 2
    sinc_convolved = aperture / lamb * np.sinc(aperture / lamb * (m.aoa_search_range - np.deg2rad(40.0)))
    plt, axs = pl.subplots(1, 2)
    axs[0].plot(cor)
    axs[1].plot(sinc_convolved)
    # pl.stem(cor, use_line_collection=True)
    pl.show()


def test_plot():
    pl.figure()
    pl.plot(2 * np.sinc(2 * np.arange(-1, 1, 0.01)))
    pl.show()
    exit()

    plt, axs = pl.subplots(2, 3)
    st = plt.suptitle('aaa', fontsize="x-large")
    for i in range(2):
        for j in range(3):
            axs[i, j].stem(np.arange(3))
            axs[i, j].set_title(f"{i},{j}")
    # pl.tight_layout()
    # shift subplots down:
    # st.set_y(0.95)
    plt.subplots_adjust(top=0.75)
    pl.show()


def test_aoa_aod_map():
    config = {'rx_antennas': 4, 'tx_antennas': 4,
              'tof_max': 100e-9, 'dop_max': 0.1,
              'initial_est_stop_threshold_db': 1.0,
              'debug': True}
    m = Mdtrack(**config)
    test1_par = np.array([[np.deg2rad(60.7), np.deg2rad(40), 20.6e-9, 0.0, 1],
                          [np.deg2rad(83.8), np.deg2rad(50), 28.1e-9, 0.0, 0.5]])
    # test1_par = np.array([[np.deg2rad(70.0), 0.0, 20.6e-9, 0.0, 0.9 * np.exp(1j * 2)]])
    sig_t = m.get_new_channel_t(test1_par, target_snr_db=None)
    print(Snr.get_avg_snr_db(sig_t, m.per_tone_noise_mw))
    exit()

    ch_est = np.fft.fft(sig_t, axis=0) * m.ltf_f[:, np.newaxis, np.newaxis]
    kwargs = {'bw': m.bw, 'fc': m.fc, 'tx_antennas': m.tx_antennas, 'rx_antennas': m.rx_antennas,
              'ant_spacing': m.ant_spacing}
    s = Signal(**kwargs)
    xx = s.get_new_channel_f(test1_par, target_snr_db=20)
    print(xx - ch_est)
    # m.show_freq_response(sig_t)
    exit()

    m.resolve_multipath(sig_t)
    hmap = np.zeros((len(m.aoa_search_range), len(m.aod_search_range)))
    for i in range(len(m.aoa_search_mat)):
        for j in range(len(m.aod_search_mat)):
            val = np.sum(m.aoa_search_mat[i].conjugate()[:, :, np.newaxis] * ch_est, axis=1)
            val = np.sum(val * m.aod_search_mat[j].conjugate()[:, :], axis=1)
            hmap[i, j] = np.linalg.norm(val)

    pl.figure()
    pl.imshow(hmap, aspect='auto', cmap=pl.get_cmap('jet'))
    pl.ylabel('AoA')
    pl.xlabel('AoD')
    pl.show()


def test_aoa_tof_map():
    config = {'rx_antennas': 4, 'tx_antennas': 1, 'tof_max': 100e-9, 'dop_max': 0.1,
              'initial_est_stop_threshold_db': 1.0, 'debug': True}
    m = Mdtrack(**config)
    test1_par = np.array([[np.deg2rad(60.7), np.deg2rad(40), 20.6e-9, 0.0, 1],
                          [np.deg2rad(83.8), np.deg2rad(50), 35.1e-9, 0.0, 0.1]])
    # test1_par = np.array([[np.deg2rad(80.0), 0.0, 20.6e-9, 0.0, 0.9 * np.exp(1j * 2)]])
    sig_t = m.get_new_channel_t(test1_par, target_snr_db=20)
    m.initial_estimation2(sig_t)
    # m.resolve_multipath(sig_t)
    # m.show_freq_response(sig_t)

    ch_est = np.fft.fft(sig_t * signal.windows.hann(64), axis=0) * m.ltf_f[:, np.newaxis, np.newaxis]
    normalized_aoa = ch_est[1:, :, 0] / np.exp(1j * np.angle(ch_est[1:, 0, 0]))[:, np.newaxis]
    normalized_tof = ch_est[:, 0, 0] / np.exp(1j * np.angle(ch_est[1, 0, 0]))
    normalized_ch = ch_est[1:, :, :] / np.exp(1j * np.angle(ch_est[1, 0, 0]))

    # m.resolve_multipath(sig_t)
    # hmap1 = np.zeros((len(m.aoa_search_range), len(m.tof_search_range)))
    # for i, aoa_mat in enumerate(m.aoa_search_mat):
    #     for j, tof in enumerate(m.tof_search_range):
    #         val = np.sum(aoa_mat.conjugate()[:, :, np.newaxis] * ch_est, axis=1)
    #         val = np.sum(val * m.aod_search_mat[0].conjugate()[:, :], axis=1)
    #         val = np.vdot(m.get_tof_effect(tof), val)
    #         hmap1[i, j] = np.linalg.norm(val)

    # normalized channel
    hmap2 = np.zeros((len(m.aoa_search_range), len(m.tof_search_range)))
    for i, aoa_mat in enumerate(m.aoa_search_mat):
        for j, tof in enumerate(m.tof_search_range):
            val = np.sum(aoa_mat.conjugate()[1:, :, np.newaxis] * normalized_ch, axis=1)
            val = np.sum(val * m.aod_search_mat[0].conjugate()[1:, :], axis=1)
            tof_vec = m.get_tof_effect(tof)
            tof_vec = tof_vec[1:] / np.exp(1j * np.angle(tof_vec[1]))
            val = np.vdot(tof_vec, val)
            hmap2[i, j] = np.linalg.norm(val)

    # # use phases only
    # hmap2 = np.zeros((len(m.aoa_search_range), len(m.tof_search_range)))
    # for i, aoa_mat in enumerate(m.aoa_search_mat):
    #     for j, tof in enumerate(m.tof_search_range):
    #         val = np.sum(aoa_mat.conjugate()[1:, :, np.newaxis] * normalized_ch, axis=1)
    #         val = np.sum(val * m.aod_search_mat[0].conjugate()[1:, :], axis=1)
    #         tof_vec = m.get_tof_effect(tof)
    #         tof_vec = tof_vec[1:] / np.exp(1j * np.angle(tof_vec[1]))
    #         val = np.vdot(tof_vec, val)
    #         hmap2[i, j] = np.linalg.norm(val)

    pl.figure()
    # plt, axs = pl.subplots(2,1)
    # axs[0].imshow(hmap1, aspect='auto', cmap=pl.get_cmap('jet'))
    pl.imshow(hmap2, aspect='auto', cmap=pl.get_cmap('jet'))
    ind = np.unravel_index(np.argmax(hmap2), hmap2.shape)
    pl.annotate(f"x ({ind[1]},{ind[0]})", xy=(ind[1], ind[0]), fontsize='large', color='w')
    print(f"aoa_deg={np.rad2deg(m.aoa_search_range[ind[0]]):.2f}, tof_ns={m.tof_search_range[ind[1]] * 1e9:.2f}")
    # pl.plot(np.arange(m.rx_antennas), np.angle(normalized_aoa.T))
    # pl.plot(np.arange(m.fftsize-1), np.angle(normalized_tof[1:]))
    # pl.ylabel('AoA')
    # pl.xlabel('AoD')
    pl.show()


def test():
    config = {'rx_antennas': 1, 'tx_antennas': 1, 'tof_max': 100e-9, 'dop_max': 0.1,
              'initial_est_stop_threshold_db': 1.0, 'preamble_repeat_cnt': 1,
              'debug': True}
    m = Mdtrack(**config)
    test1_par = np.array([[np.deg2rad(0.0), np.deg2rad(0), 60.0e-9, 0.0, 1],
                          [np.deg2rad(83.8), np.deg2rad(50), 28.1e-9, 0.0, 0.1]])
    sig_t = m.get_new_channel_t(test1_par, target_snr_db=15)
    sig_pwr = Snr.get_avg_sig_pwr(sig_t)
    print(sig_pwr)
    m.resolve_multipath(sig_t[np.newaxis,:,:,:])
    # ch_est_f = np.fft.fft(sig_t, axis=0) * m.ltf_f[:, np.newaxis, np.newaxis]
    # ch_est_t = np.fft.ifft(ch_est_f[:, 0, 0])
    # pl.figure()
    # pl.stem(np.abs(ch_est_t))
    # pl.show()


if __name__ == '__main__':
    test_correctness_and_timer()
    # test_detect_doppler_shift()
    # test_detect_doppler_shift_spaced_preambles()
    # test_tof_correlation()
    # test_aoa_correlation()
    # test_plot()
    # test_aoa_aod_map()
    # test_aoa_tof_map()
    # test()
