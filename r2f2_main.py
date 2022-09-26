from r2f2_sigcomm16.core import R2f2
from mdtrack_mobicom19.core import Mdtrack
from utils.common import timer
import numpy as np
from scipy.optimize import minimize, rosen, rosen_der, rosen_hess
from scipy.ndimage.filters import maximum_filter
from ipopt import minimize_ipopt, setLoggingLevel
import logging
import matplotlib.pyplot as pl
from scipy.special import diric
from scipy import ndimage, signal

logger = logging.getLogger(__name__)


def test_r2f2():
    @timer
    def run_mdtrack(m, sig_t):
        return m.resolve_multipath(sig_t)

    @timer
    def run_r2f2(r, sig_t):
        return r.resolve_multipath(sig_t)

    config = {'rx_antennas': 4, 'tx_antennas': 1,
              'tof_max': 250e-9, 'dop_max': 0.1, 'debug': True,
              'initial_est_stop_threshold_db': 1.0}
    m = Mdtrack(**config)
    r = R2f2(**config)
    test_par = np.array([[np.deg2rad(60.7), np.deg2rad(40), 20.6e-9, 0.0, 1],
                         [np.deg2rad(73.8), np.deg2rad(50), 28.1e-9, 0.0, 0.5]])
    # test_par = np.array([[np.deg2rad(70.7), np.deg2rad(0), 125e-9, 0.0, 1],
    #                       [np.deg2rad(120), np.deg2rad(0), 65.1e-9, 0.0, 0.8 ]])
    # test_par = np.array([[np.deg2rad(40.0), 0.0, 20.6e-9, 0.0, 0.9 * np.exp(1j * 2)]])
    sig_t = m.get_new_channel_t(test_par, target_snr_db=20)
    H = np.fft.fft(sig_t, axis=0) * m.ltf_f[:, np.newaxis, np.newaxis]
    hmap = r.get_aoa_tof_heatmap(H[:,:,0])
    peaks = r.filter_peaks(r.get_ranked_peaks(hmap, debug=True))
    r.show_aoa_tof_heatmap(hmap, peaks)
    v_list = r.resolve_multipath(sig_t)
    print(r.print_pretty(v_list))
    exit()

    # m.initial_estimation(sig_t)
    # m.show_heatmap()
    # m.single_path_estimation(sig_t, plot=True)
    # m.resolve_multipath(sig_t)
    # res1, runtime1 = run_mdtrack(m, sig_t)
    res2, runtime2 = run_r2f2(r, sig_t)
    print(r.print_pretty(res2))

    # f1 = a.rx_F[0,:,:]
    # f2 = a.rx_F_inv[0,:,:]
    # S = a.rx_S[:, :, [1,2]] * a.D[:, np.newaxis, [3,4]]
    # print(f1@f2@np.arange(3))
    # print(f1)
    # print(f2)
    # print(f1@f2)
    # print(f2@f1)


def test_dirichlet_aoa():
    np.set_printoptions(suppress=True)

    def dirichlet(w, N, center=0):
        return np.exp(1j * (w - center) * (N - 1) / 2) * diric(w - center, N)

    # a = np.random.rand(4)
    # b = np.random.rand(2)
    # print(dirichlet(a,4,b).shape)
    # exit()

    config = {'rx_antennas': 8, 'tx_antennas': 1,
              'tof_max': 100e-9, 'dop_max': 0.1, 'debug': True,
              'initial_est_stop_threshold_db': 1.0}
    r = R2f2(**config)

    min_spaing = 3e8 / r.fc / 2 / r.wavelength[len(r.wavelength) // 2]
    max_spaing = 3e8 / r.fc / 2 / r.wavelength[len(r.wavelength) // 2 - 1]
    print(f"min antenna spacing={min_spaing}")
    print(f"max antenna spacing={max_spaing}")

    rx_antennas = 8
    fftpoints = rx_antennas * 16
    ant_spacing1 = 0.5  # in unit of lambda
    ant_spacing2 = np.round(min_spaing, 5)  # in unit of lambda
    ground_truth_aoa = 0.66  # given as cos value
    # different wavelength gives different array response
    array_res_1 = np.exp(-1j * 2 * np.pi * np.arange(rx_antennas) * ant_spacing1 * ground_truth_aoa)
    array_res_2 = np.exp(-1j * 2 * np.pi * np.arange(rx_antennas) * ant_spacing2 * ground_truth_aoa)
    plt, axs = pl.subplots(3, 2)
    axs[0, 0].plot(np.angle(array_res_1))
    axs[0, 0].plot(np.angle(array_res_2))
    axs[0, 0].legend([f"ant_spacing={ant_spacing1}", f"ant_spacing={ant_spacing2}"])
    axs[0, 0].set_ylabel('Phase')
    axs[0, 0].set_xlabel('antenna')

    cos_vals = np.linspace(-1, 1, fftpoints, endpoint=False)
    fourier_inv_1 = 1 / rx_antennas * np.exp(
        1j * 2 * np.pi * np.arange(rx_antennas) * ant_spacing1 * cos_vals[:, np.newaxis])
    fourier_inv_2 = 1 / rx_antennas * np.exp(
        1j * 2 * np.pi * np.arange(rx_antennas) * ant_spacing2 * cos_vals[:, np.newaxis])
    ifft1 = fourier_inv_1 @ array_res_1
    # ifft1 = r.rx_F_inv[0,:,:] @ array_res_1
    ifft2 = fourier_inv_2 @ array_res_2
    axs[1, 0].plot(np.abs(ifft1))
    axs[1, 0].plot(np.abs(ifft2))
    axs[1, 0].legend([f"ant_spacing={ant_spacing1}", f"ant_spacing={ant_spacing2}"])
    axs[1, 0].set_ylabel('Magitude')
    axs[1, 0].set_xlabel('cos theta')
    xtickpos = np.linspace(0, fftpoints, 5, endpoint=False)
    axs[1, 0].set_xticks(xtickpos)
    axs[1, 0].set_xticklabels([f"{cos_vals[int(x)]:.1f}" for x in xtickpos])

    axs[1, 1].plot(np.angle(ifft1))
    axs[1, 1].plot(np.angle(ifft2))
    axs[1, 1].legend([f"ant_spacing={ant_spacing1}", f"ant_spacing={ant_spacing2}"])
    axs[1, 1].set_ylabel('Phase')
    axs[1, 1].set_xlabel('cos theta')
    axs[1, 1].set_xticks(xtickpos)
    axs[1, 1].set_xticklabels([f"{cos_vals[int(x)]:.1f}" for x in xtickpos])

    # inputs for dirichlet
    w1 = 2 * np.pi * ant_spacing1 * cos_vals  # radian per space sample
    w2 = 2 * np.pi * ant_spacing2 * cos_vals  # radian per space sample
    center1 = 2 * np.pi * ant_spacing1 * ground_truth_aoa
    center2 = 2 * np.pi * ant_spacing2 * ground_truth_aoa
    diri1 = dirichlet(w1, rx_antennas, center1)
    diri2 = dirichlet(w2, rx_antennas, center2)
    axs[2, 0].plot(np.abs(diri1))
    axs[2, 0].plot(np.abs(diri2))
    axs[2, 0].legend([f"ant_spacing={ant_spacing1}", f"ant_spacing={ant_spacing2}"])
    axs[2, 0].set_ylabel('Magitude')
    axs[2, 0].set_xlabel('cos theta')
    axs[2, 0].set_xticks(xtickpos)
    axs[2, 0].set_xticklabels([f"{cos_vals[int(x)]:.1f}" for x in xtickpos])

    axs[2, 1].plot(np.angle(diri1))
    axs[2, 1].plot(np.angle(diri2))
    axs[2, 1].legend([f"ant_spacing={ant_spacing1}", f"ant_spacing={ant_spacing2}"])
    axs[2, 1].set_ylabel('Phase')
    axs[2, 1].set_xlabel('cos theta')
    axs[2, 1].set_xticks(xtickpos)
    axs[2, 1].set_xticklabels([f"{cos_vals[int(x)]:.1f}" for x in xtickpos])

    print(f"ground truth aoa (cos value): {ground_truth_aoa}")
    print(f"aoa from ifft with antenna spacing {ant_spacing1} (lambda): {cos_vals[np.argmax(np.abs(ifft1))]} ")
    print(f"aoa from ifft with antenna spacing {ant_spacing2} (lambda): {cos_vals[np.argmax(np.abs(ifft2))]} ")
    print(f"aoa from dirichlet with antenna spacing {ant_spacing1} (lambda): {cos_vals[np.argmax(np.abs(diri1))]} ")
    print(f"aoa from dirichlet with antenna spacing {ant_spacing2} (lambda): {cos_vals[np.argmax(np.abs(diri2))]} ")
    print(f"antenna spacing {ant_spacing1} norm(ifft1 - dirichlet1)={np.linalg.norm(ifft1 - diri1)}")
    print(f"antenna spacing {ant_spacing2} norm(ifft2 - dirichlet2)={np.linalg.norm(ifft2 - diri2)}")
    print(f"antenna spacing {ant_spacing2} norm(ifft2 - dirichlet1)={np.linalg.norm(ifft2 - diri1)}")
    print(f"The larger the array size or bandwidth, the greater the error accumulated.")
    pl.show()

def test_ifft():
    subcarriers = 16
    tof1, tof2 = 1, 2.5
    amp1, amp2 = 1, 0.1
    sig1 = amp1 * np.exp(-1j*2*np.pi*np.arange(subcarriers)*tof1/subcarriers)
    sig2 = np.exp(-1j*2*np.pi*np.arange(subcarriers)*tof2/subcarriers)
    pl.figure()
    pl.plot(np.abs(np.fft.ifft(sig1, 256)), label='amp=1')
    pl.plot(np.abs(np.fft.ifft(sig2/10, 256)), label='amp=1/10')
    pl.plot(np.abs(np.fft.ifft(sig2/20, 256)), label='amp=1/20')
    pl.plot(np.abs(np.fft.ifft(sig2/30, 256)), label='amp=1/30')
    pl.legend()
    pl.show()


def test_dirichlet_tof():
    np.set_printoptions(suppress=True)

    def dirichlet(w, N, center=0):
        return np.exp(1j * (w - center) * (N - 1) / 2) * diric(w - center, N)

    # a = np.random.rand(4)
    # b = np.random.rand(2)
    # print(dirichlet(a,4,b).shape)
    # exit()

    bw = 20e6
    tofs = np.array([425e-9])
    tofs_in_samp = np.array([5.3, 16]) #tofs * bw
    fftpoints = 32

    sig_f = np.zeros(fftpoints, dtype=np.complex)
    for tof_in_samp in tofs_in_samp:
        sig_f += np.exp(-1j * 2 * np.pi * np.arange(fftpoints) / fftpoints * tof_in_samp)
    sig_t = np.fft.ifft(sig_f, fftpoints)

    w = 2 * np.pi * np.arange(fftpoints) / fftpoints  # radian per time sample
    centers = 2 * np.pi / fftpoints * tofs_in_samp

    diri = np.zeros(fftpoints, dtype=np.complex)
    for center in centers:
        diri += dirichlet(w, fftpoints, center)

    plt, axs = pl.subplots(2, 2)
    axs[0, 0].plot(np.abs(sig_t))
    axs[0, 0].legend(['ifft'])
    axs[0, 0].set_ylabel('Magitude')

    axs[1, 0].plot(np.abs(diri))
    axs[1, 0].legend(['dirichlet'])
    axs[1, 0].set_ylabel('Magitude')

    axs[0, 1].plot(np.angle(sig_t))
    axs[0, 1].legend(['ifft'])
    axs[0, 1].set_ylabel('Phase')

    axs[1, 1].plot(np.angle(diri))
    axs[1, 1].legend(['dirichlet'])
    axs[1, 1].set_ylabel('Phase')
    pl.show()

    print(f"norm(ifft - dirichlet)={np.linalg.norm(sig_t - diri)}")

def test_dirichlet_doppler():
    np.set_printoptions(suppress=True)

    def dirichlet(w, N, center=0):
        return np.exp(1j * (w - center) * (N - 1) / 2) * diric(w - center, N)

    # # 1. test two frequency components
    # bw = 20e6
    # fftpoints = 32
    # subcarrier_spacing = bw/fftpoints # Hz
    # doppler = 0.3 * subcarrier_spacing # Hz
    # freq_contents = np.array([7, 13]) * subcarrier_spacing + doppler # Hz
    # freq_contents_in_sample = freq_contents / subcarrier_spacing
    # complex_gains = [0.8*np.exp(1j*2), 0.5*np.exp(1j*0.9)]
    #
    # w = 2 * np.pi * np.arange(fftpoints) / (fftpoints)  # radian per time sample
    # centers = 2 * np.pi / fftpoints * freq_contents_in_sample
    # diri = np.zeros(fftpoints, dtype=np.complex)
    # for cgain, center in zip(complex_gains, centers):
    #     diri += cgain * np.conjugate(dirichlet(w, fftpoints, center))
    #
    # sig_f = np.zeros(fftpoints, dtype=np.complex)
    # for cgain, freq_content in zip(complex_gains, freq_contents_in_sample):
    #     integer_part = int(freq_content)
    #     fractional_part = freq_content - integer_part
    #     print(f"freq in sample: (integral {integer_part}), (fractional {fractional_part})")
    #     tmp_sig_f = np.zeros(fftpoints, dtype=np.complex)
    #     tmp_sig_f[integer_part] = cgain
    #     sig_t = np.fft.ifft(tmp_sig_f, fftpoints) * np.exp(1j*2*np.pi*(fractional_part*subcarrier_spacing/bw)*(np.arange(fftpoints)))
    #     sig_f += np.fft.fft(sig_t)
    #
    # plt, axs = pl.subplots(2, 2)
    # axs[0, 0].stem(np.abs(sig_f), use_line_collection=True)
    # axs[0, 0].legend(['fft'])
    # axs[0, 0].set_ylabel('Magitude')
    #
    # axs[1, 0].stem(np.abs(diri), use_line_collection=True)
    # axs[1, 0].legend(['dirichlet'])
    # axs[1, 0].set_ylabel('Magitude')
    #
    # axs[0, 1].plot(np.angle(sig_f))
    # axs[0, 1].legend(['fft'])
    # axs[0, 1].set_ylabel('Phase')
    #
    # axs[1, 1].plot(np.angle(diri))
    # axs[1, 1].legend(['dirichlet'])
    # axs[1, 1].set_ylabel('Phase')
    # pl.show()
    #
    # print(f"norm(fft - dirichlet)={np.linalg.norm(sig_f - diri)}")


    # # 2. test a slope response
    # bw = 40e6
    # fftpoints = 32
    # observation_points = fftpoints # has to be the same as fftpoints
    # subcarrier_spacing = bw / fftpoints  # Hz
    # doppler = 18#0.0006 * subcarrier_spacing  # Hz
    # freq_contents = np.arange(fftpoints) * subcarrier_spacing + doppler  # Hz
    # freq_contents_in_sample = freq_contents / subcarrier_spacing
    # tof = 5  # sample delay
    # complex_gains = np.exp(-1j*2*np.pi*tof/fftpoints*np.arange(fftpoints))
    #
    # w = 2 * np.pi * np.arange(observation_points) / observation_points  # radian per time sample
    # centers = 2 * np.pi / fftpoints * freq_contents_in_sample
    # diri = np.zeros(observation_points, dtype=np.complex)
    # for cgain, center in zip(complex_gains, centers):
    #     diri += cgain * np.conjugate(dirichlet(w, observation_points, center))
    #
    # sig_f = complex_gains
    # sig_t = np.fft.ifft(sig_f, fftpoints) * np.exp(1j * 2 * np.pi * (doppler / bw) * (np.arange(fftpoints)))
    # sig_f = np.fft.fft(sig_t, observation_points)
    #
    # plt, axs = pl.subplots(2, 2)
    # axs[0, 0].stem(np.abs(sig_f), use_line_collection=True)
    # axs[0, 0].legend(['fft'])
    # axs[0, 0].set_ylabel('Magitude')
    #
    # axs[1, 0].stem(np.abs(diri), use_line_collection=True)
    # axs[1, 0].legend(['dirichlet'])
    # axs[1, 0].set_ylabel('Magitude')
    #
    # axs[0, 1].plot(np.angle(sig_f))
    # axs[0, 1].legend(['fft'])
    # axs[0, 1].set_ylabel('Phase')
    #
    # axs[1, 1].plot(np.angle(diri))
    # axs[1, 1].legend(['dirichlet'])
    # axs[1, 1].set_ylabel('Phase')
    # pl.show()
    #
    # print(f"norm(fft - dirichlet)={np.linalg.norm(sig_f - diri)}")


    from mdtrack_mobicom19.core import Mdtrack
    m = Mdtrack({'bw':40e6, 'tx_antennas': 1, 'rx_antennas':1, 'preamble_repeat_cnt':1})
    test1_par = np.array([[np.deg2rad(70.7), np.deg2rad(0), 17.6e-9, 0.0, 0.8],
                          [np.deg2rad(110), np.deg2rad(0), 55.1e-9, 18.0, 0.8 / 20]])
    sig_t = m.get_new_channel_t(test1_par, target_snr_db=25)
    sig_f = np.fft.fft(sig_t, axis=0)
    H = sig_f * m.ltf_f[:, np.newaxis, np.newaxis]
    H = H[:,0,0]

    # pl.figure()
    # pl.plot(abs(np.fft.ifft(H)))
    # pl.show()
    # exit()

    fftpoints = 64
    dop_shifts = np.arange(0.0,20,1)
    tof_ns = np.arange(0,100,1)*1e-9
    tof_sample_delay = tof_ns * m.bw
    cost = np.zeros((len(dop_shifts), len(tof_sample_delay)))
    for i, dop in enumerate(dop_shifts):
        for k, tof in enumerate(tof_sample_delay):
            complex_gains = np.exp(-1j*2*np.pi*tof/fftpoints*np.arange(fftpoints))
            w = 2 * np.pi * np.arange(fftpoints) / fftpoints
            centers = 2 * np.pi / fftpoints * np.arange(fftpoints) + 2 * np.pi *dop/m.bw
            diri = np.zeros(fftpoints, dtype=np.complex)
            for cgain, center in zip(complex_gains, centers):
                diri += cgain * np.conjugate(dirichlet(w, fftpoints, center))
            cost[i,k] = np.linalg.norm(H - diri)
    pl.figure()
    pl.imshow(cost, aspect='auto', cmap=pl.get_cmap('jet'))
    pl.tight_layout()
    pl.colorbar()
    pl.show()

    trough = np.where(cost == np.min(cost))
    print(f"dop = {dop_shifts[trough[0][0]]}, tof={tof_sample_delay[trough[1][0]]/m.bw}")

def test_dirichlet_width():
    np.set_printoptions(suppress=True)

    def dirichlet(w, N, center=0):
        return np.exp(1j * (w - center) * (N - 1) / 2) * diric(w - center, N)

    # plt, axs = pl.subplots(2,1)
    # x = signal.windows.gaussian(64, 7)
    # y = 20*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x, 1024))))
    # axs[0].plot(x)
    # axs[1].plot(np.linspace(-0.5, 0.5, 1024), y)
    # pl.show()
    # exit()

    # pl.figure()
    # x = np.linspace(-0.5, 0.5, 512)
    # pl.plot(x, np.abs(np.fft.fftshift(np.fft.fft(signal.windows.gaussian(64, 32), 512))))
    # # pl.plot(x, np.abs(np.fft.fftshift(np.fft.fft(signal.windows.gaussian(128, 64), 512))))
    # pl.plot(x, np.abs(np.fft.fftshift(np.fft.fft(signal.windows.hann(64), 512))))
    # pl.plot(x, np.abs(np.fft.fftshift(np.fft.fft(np.ones(64), 512))))
    # pl.legend(['gaussian', 'hann', 'boxcar'])
    # pl.xlim([-0.1, 0.1])
    # pl.show()
    # exit()

    super_resolution_ratio = 0.9
    sigma = super_resolution_ratio * 30 / 3
    subcarriers = 64
    fftpoints = 64 * 3 * 10
    peak1, peak2 = 19.0, 20.5
    # peak1, peak2 = 20.6/50, 28.1/50
    sig_f1 = np.exp(-1j * 2 * np.pi * np.arange(subcarriers) / subcarriers * peak1)
    sig_f2 = np.exp(-1j * 2 * np.pi * np.arange(subcarriers) / subcarriers * peak2)
    sig_f = sig_f1 + sig_f2
    sig_t = np.fft.ifft(sig_f, fftpoints) * fftpoints / subcarriers

    win = signal.windows.gaussian(subcarriers, 0.9*64)
    sig_t_windowed = np.fft.ifft(sig_f * win, fftpoints) * fftpoints / subcarriers
    # pl.figure()
    # pl.plot(win)
    # pl.plot(np.abs(np.fft.fft(win)))
    # pl.show()
    # exit()

    w = 2 * np.pi * np.arange(fftpoints) / fftpoints  # radian per time sample
    center1 = 2 * np.pi / subcarriers * peak1
    center2 = 2 * np.pi / subcarriers * peak2
    diri1 = dirichlet(w, subcarriers, center1)
    diri2 = dirichlet(w, subcarriers, center2)
    diri = diri1 + diri2

    optml = np.zeros(fftpoints)
    optml[int(peak1 * fftpoints // 64)] = 1.0
    optml[int(peak2 * fftpoints // 64)] = 1.0
    optml = ndimage.gaussian_filter1d(optml, sigma) * np.sqrt(2 * np.pi) * sigma

    # print(signal.find_peaks(y))
    pl.figure()
    # pl.stem(np.abs(sig_t))  # sig_t == diri
    pl.plot(np.abs(diri), 'k')
    pl.plot(np.abs(diri1), 'r')
    pl.plot(np.abs(diri2), 'g')
    pl.plot(np.abs(sig_t_windowed), 'y')
    pl.stem(np.abs(optml))
    print('This shows optml has higher resolution than just using gaussian windowing function.')
    pl.legend(['sum_dirichelet', 'dirichet1', 'dirichet2', 'windowed_ifft', 'optml'])
    pl.xlim([(peak1 - 2) * fftpoints // 64, (peak2 + 2) * fftpoints // 64])
    pl.show()
    exit()


def test_dirichlet_width_2d():
    np.set_printoptions(suppress=True)

    def dirichlet(w, N, center=0):
        return np.exp(1j * (w - center) * (N - 1) / 2) * diric(w - center, N)

    def get_tof(tof_peak, subcarriers):
        return np.exp(-1j * 2 * np.pi * np.arange(subcarriers) / subcarriers * tof_peak)

    def get_aoa(aoa_peak, rx_antennas):
        return np.exp(-1j * 2 * np.pi * 0.5 * np.arange(rx_antennas) * aoa_peak)

    def get_2d_heatmap(sig_f, subcarriers, rx_antennas, tof_fftpoints, aoa_fftpoints):
        tofs = np.linspace(0, subcarriers, tof_fftpoints, endpoint=False)
        aoas = np.fft.fftshift(np.linspace(-1, 1, aoa_fftpoints, endpoint=False))
        hmap = np.zeros((len(aoas), len(tofs)))
        for i, aoa in enumerate(aoas):
            for j, tof in enumerate(tofs):
                cor = get_tof(tof, subcarriers)[:, np.newaxis] * get_aoa(aoa, rx_antennas)[np.newaxis, :]
                hmap[j, i] = np.abs(np.sum(sig_f * cor.conjugate())) ** 2
        return hmap

    def detect_peaks(image):
        """
        A code snippet from stackoverflow.
        Takes an image and detect the peaks usingthe local maximum filter.
        Returns a boolean mask of the peaks (i.e. 1 when the pixel's value is the neighborhood maximum, 0 otherwise)
        """

        threshold = 0.01
        window = (5, 5)
        local_max = maximum_filter(image, size=window, mode='wrap')
        new_image = image / local_max
        detected_peaks = np.logical_and(image >= threshold, new_image == 1)

        plt, axs = pl.subplots(3, 1)
        axs[0].imshow(image, aspect='auto', cmap=pl.get_cmap('jet'))
        axs[1].imshow(new_image, aspect='auto', cmap=pl.get_cmap('jet'))
        axs[2].imshow(detected_peaks, aspect='auto', cmap=pl.get_cmap('jet'))
        pl.show()
        # exit()

        # # define an 8-connected neighborhood
        # neighborhood = generate_binary_structure(2, 2)
        #
        # # apply the local maximum filter; all pixel of maximal value
        # # in their neighborhood are set to 1
        # local_max = maximum_filter(image, footprint=neighborhood) == image
        # # local_max = maximum_filter(image, size=(5,5)) == image
        # # local_max is a mask that contains the peaks we are
        # # looking for, but also the background.
        # # In order to isolate the peaks we must remove the background from the mask.
        #
        # # we create the mask of the background
        # background = (image == 0)
        #
        # # a little technicality: we must erode the background in order to
        # # successfully subtract it form local_max, otherwise a line will
        # # appear along the background border (artifact of the local maximum filter)
        # eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
        #
        # # we obtain the final mask, containing only peaks,
        # # by removing the background from the local_max mask (xor operation)
        # detected_peaks = local_max ^ eroded_background

        return detected_peaks

    super_resolution_ratio = 1.0
    sigma = super_resolution_ratio * 30 / 3
    subcarriers = 4
    rx_antennas = 4
    tof_fftpoints = subcarriers * 3 * 10
    aoa_fftpoints = rx_antennas * 3 * 10
    # tof_peak1, tof_peak2 = 1.0, 1.5
    # aoa_peak1, aoa_peak2 = 0.4, 0.7  # -1<= cos <= 1
    tof_peak1, tof_peak2 = 17.6/50, 120.1/50
    aoa_peak1, aoa_peak2 = np.cos(np.deg2rad(60.7)), np.cos(np.deg2rad(123.8))  # -1<= cos <= 1
    amp1, amp2 = 1.0, 0.05

    sig_f1 = get_tof(tof_peak1, subcarriers)[:, np.newaxis] * get_aoa(aoa_peak1, rx_antennas)[np.newaxis, :]
    sig_f2 = get_tof(tof_peak2, subcarriers)[:, np.newaxis] * get_aoa(aoa_peak2, rx_antennas)[np.newaxis, :]
    sig_f = amp1*sig_f1 + amp2*sig_f2
    hmap = get_2d_heatmap(sig_f, subcarriers, rx_antennas, tof_fftpoints, aoa_fftpoints)
    # sig_t = np.fft.ifft(sig_f, fftpoints) * fftpoints / subcarriers
    #
    # win = signal.windows.gaussian(subcarriers, subcarriers // 3)
    # sig_t_windowed = np.fft.ifft(sig_f * win, fftpoints) * fftpoints / subcarriers

    w_tofs = 2 * np.pi / tof_fftpoints * np.arange(tof_fftpoints)
    w_aoas = np.fft.fftshift(2 * np.pi * 0.5 * np.linspace(-1, 1, aoa_fftpoints, endpoint=False))
    # w = 2 * np.pi * np.arange(fftpoints) / fftpoints  # radian per time sample
    tof_center1 = 2 * np.pi / subcarriers * tof_peak1
    tof_center2 = 2 * np.pi / subcarriers * tof_peak2
    aoa_center1 = 2 * np.pi * 0.5 * aoa_peak1
    aoa_center2 = 2 * np.pi * 0.5 * aoa_peak2

    diri_tof1 = dirichlet(w_tofs, subcarriers, tof_center1)
    diri_tof2 = dirichlet(w_tofs, subcarriers, tof_center2)
    diri_tof = diri_tof1 + diri_tof2

    diri_aoa1 = dirichlet(w_aoas, rx_antennas, aoa_center1)
    diri_aoa2 = dirichlet(w_aoas, rx_antennas, aoa_center2)
    diri_aoa = diri_aoa1 + diri_aoa2

    # diri = np.abs(diri_aoa[:, np.newaxis])**2 * np.abs(diri_tof)**2
    # diri = np.abs(diri_tof[:, np.newaxis] * diri_aoa[np.newaxis,:])**2
    # diri = np.abs(diri_tof[:, np.newaxis]*np.ones(len(diri_aoa))[np.newaxis, :] +
    #               diri_aoa[np.newaxis,:]*np.ones(len(diri_tof))[:, np.newaxis])**2
    diri = np.abs(amp1 * diri_tof1[:, np.newaxis] * diri_aoa1[np.newaxis, :] +
                  amp2 * diri_tof2[:, np.newaxis] * diri_aoa2[np.newaxis, :]) ** 2

    optml = np.zeros((tof_fftpoints, aoa_fftpoints))
    optml[int(tof_peak1 * tof_fftpoints // subcarriers), int((aoa_peak1+1) * aoa_fftpoints / 2)] = amp1
    optml[int(tof_peak2 * tof_fftpoints // subcarriers), int((aoa_peak2+1) * aoa_fftpoints / 2)] = amp2
    optml = ndimage.gaussian_filter(optml, sigma) * 2 * np.pi * sigma**2

    # print(signal.find_peaks(y))
    plt, axs = pl.subplots(2,2)
    axs[0, 0].imshow(hmap, aspect='auto', cmap=pl.get_cmap('jet'))
    axs[0, 0].set_title('2D correlation')
    axs[0, 0].set_xlabel('AoA')
    axs[0, 0].set_ylabel('ToF')
    # axs[1].imshow(np.abs(diri), aspect='auto', cmap=pl.get_cmap('jet'))
    axs[0, 1].imshow(np.abs(np.fft.ifft2(sig_f, s=(tof_fftpoints, aoa_fftpoints)))**2, aspect='auto', cmap=pl.get_cmap('jet'))
    axs[0, 1].set_title('2D ifft')
    axs[1, 0].imshow(diri, aspect='auto', cmap=pl.get_cmap('jet'))
    axs[1, 0].set_title('2D dirichlet')
    axs[1, 1].imshow(optml, aspect='auto', cmap=pl.get_cmap('jet'))
    axs[1, 1].set_title('2D optml')
    print(np.where(detect_peaks(hmap) == True))
    print(np.where(detect_peaks(optml) == True))
    # axs[2].plot(diri_tof)
    # axs[3].plot(diri_aoa)

    # pl.stem(np.abs(sig_t))  # sig_t == diri
    # pl.plot(np.abs(diri), 'k')
    # pl.plot(np.abs(diri1), 'r')
    # pl.plot(np.abs(diri2), 'g')
    # pl.plot(np.abs(sig_t_windowed), 'y')
    # pl.stem(np.abs(optml))
    # print('This shows optml has higher resolution than just using gaussian windowing function.')
    # pl.legend(['sum_dirichelet', 'dirichet1', 'dirichet2', 'windowed_ifft', 'optml'])
    # pl.xlim([(peak1 - 2) * fftpoints // 64, (peak2 + 2) * fftpoints // 64])
    pl.tight_layout()
    pl.show()
    exit()


def scipy_solve_optimization():
    def objective_func(x, *args):
        print(x)
        center = [1, 2, 3, 4]
        N = args[0]
        res = 0
        for i, val in enumerate(center[:N]):
            res += (x[i] - val) ** 2
        return res

    @timer
    def ipopt_solve(func, x0, bnds):
        return minimize_ipopt(func, x0, args=(4,), bounds=bnds)

    @timer
    def powell_solve(func, x0, bnds=None):
        return minimize(func, x0, args=(4,), method='Powell', bounds=bnds)

    setLoggingLevel(logging.ERROR)
    x0 = np.array([5, 6, 7, 8])
    bnd = ((0, 10), (0, 10), (0, 10), (0, 10))
    res1 = powell_solve(objective_func, x0)
    print(res1)
    # res2 = ipopt_solve(objective_func, x0, bnd)
    # print(res2)


def for_fun():
    @timer
    def a(x):
        y = 0
        for i in range(len(x)):
            y += x[i]
        return y

    @timer
    def b(x):
        y = 0
        for idx, val in enumerate(x):
            y += val
        return val

    from functools import lru_cache
    @lru_cache
    def cc(x):
        return np.exp(1j * x)

    # x = [i for i in range(1000000)]
    # a(x)
    # b(x)
    print(cc.cache_info())
    cc(1)
    cc(1)
    cc(1)
    print(cc.cache_info())


if __name__ == '__main__':
    test_r2f2()
    # test_dirichlet_aoa()
    # test_dirichlet_tof()
    # test_dirichlet_doppler()
    # test_ifft()
    # test_dirichlet_width()
    # test_dirichlet_width_2d()
    # scipy_solve_optimization()
    # for_fun()
