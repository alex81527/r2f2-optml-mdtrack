import sys
sys.path.append("../")

# if 'DISPLAY' not in os.environ:
#     import matplotlib
#     matplotlib.use('Agg')

import time, keras, json, collections, operator, os, bisect
import matplotlib.pyplot as pl
import numpy as np
import scipy.io as sio
from common_modules import r2f2_helper, r2f2, dataset_gen, rf_common
from mdtrack_mobicom19 import core
from tabulate import tabulate
from scipy import ndimage
# from skimage.feature import peak_local_max
def ski_peak(model, channel, distances,psis):
    nrows = len(psis)
    ncols = len(distances)
    ch = channel.transpose().ravel()
    ch = np.concatenate([np.real(ch), np.imag(ch)])
    pred = model.predict(ch.reshape(1,-1))
    pred = pred.reshape(nrows, ncols)
    
    # coordinates = peak_local_max(pred, min_distance=2)
    # print(coordinates)
    th = ndimage.filters.maximum_filter(pred, 5)
    filt = np.where(pred<th, 0, pred)

    pl.figure()
    pl.imshow(filt, aspect='auto')
    # pl.plot(coordinates[:, 1], coordinates[:, 0], 'rx')
    pl.show()
    exit()

def get_ml_guesses(model, channel, distances,psis,show_plots=False):
    nrows = len(psis)
    ncols = len(distances)
    ch = channel.transpose().ravel()
    ch = np.concatenate([np.real(ch), np.imag(ch)])
    pred = model.predict(ch.reshape(1,-1))
    pred = pred.reshape(nrows, ncols)

    if show_plots:
        pl.subplot(2, 2, 1)
        pl.imshow(pred, aspect='auto',cmap=pl.get_cmap('jet'))

    # p_d_psi = r2f2_helper.get_peaks(pred, distances, psis, show_plots=True)
    psi_d_mat = pred
    win = 5
	
    y = np.array(psi_d_mat)
	
    th = ndimage.filters.gaussian_filter(y, win)
    y = np.where(y<th, 0, y)
    if show_plots:
        pl.subplot(2,2,2)
        pl.imshow(y,aspect='auto',cmap=pl.get_cmap('jet'))
        pl.colorbar()

	
    filt = ndimage.filters.maximum_filter(y, win)
    th = np.mean(y)
    filt = np.where(filt<=th, th, filt)
	
    y = y/filt

    if show_plots:
        pl.subplot(2,2,3)
        pl.imshow(y,aspect='auto',cmap=pl.get_cmap('jet'))
        pl.colorbar()

    th = 1
    y = np.where(y>=th,1,0)

    if show_plots:
        pl.subplot(2,2,4)
        pl.imshow(y,aspect='auto',cmap=pl.get_cmap('jet'))
        pl.colorbar()
        pl.show()
	
    i_psis, i_distances = np.where(y==1)

    new_p_d_psi = {}
    for i in range(len(i_psis)):
        key = (distances[i_distances[i]], psis[i_psis[i]])
        new_p_d_psi[key] = psi_d_mat[i_psis[i], i_distances[i]]
    p_d_psi = new_p_d_psi

    ####sort guesses by "probability"
    p_d_psi = sorted(list(p_d_psi.items()),
                     key=operator.itemgetter(1), reverse=True)
    # print(p_d_psi)
    p_d_psi = [k for k, v in p_d_psi]
    temp = []
    list(map(temp.extend, p_d_psi))
    p_d_psi = temp

    return p_d_psi

def beamforming_snr(ch_l2, ch_l2_guess):
    return np.mean(np.abs(np.sum(ch_l2/(ch_l2_guess/np.abs(ch_l2_guess)),axis=1))**2)

def get_loc_error(ground, pred):
    ground_ds, ground_psis = ground[0], ground[3]
    pred_ds, pred_psis = pred[0], pred[3]
    idx1, idx2 = np.argmin(ground_ds), np.argmin(pred_ds)
    d1, d2 = ground_ds[idx1], pred_ds[idx2]
    theta1, theta2 = np.arccos(ground_psis[idx1]), np.arccos(pred_psis[idx2])
    return np.linalg.norm(d1*np.exp(1j*theta1) - d2*np.exp(1j*theta2))

if __name__ == '__main__':
    # ML parameter
    data_from_pickle = False
    num_chans = int(5e5)
    num_chans_test = int(num_chans/10)
    min_d, max_d = 0, 100
    d_step, psi_step = 4, 0.1 # NN output granularity
    d_sep, aoa_sep = 8, 0     # training data constraints, ToF (meter) AoA (rad)
    distances = np.arange(min_d, max_d, d_step)
    psis = np.arange(-1, 1, psi_step) # cos values, -1 to 1 for bisect
    ncols, nrows = len(distances), len(psis)
    sigma = 1.0
    norm = True
    n_cores = 8

    # RF parameters
    K = 4 # antenna count
    min_n_paths = 1
    max_n_paths = 4
    lowest_amp = 0.05
    bw = 20e6
    nfft = 64    
    cf1 = 2.412e9
    cf2 = cf1+30e6
    K2 = K
    l1 = rf_common.get_lambs(cf1, bw, nfft)
    l2 = rf_common.get_lambs(cf2, bw, nfft)
    sep = l1[0]/2
    noise_pwr = 1e-5
    
    # S = r2f2.get_SD(L=K*sep, K=K, N=2, all_lambs=l1, d_ns=np.array([7,12]), psi_js=np.array([0.48,0.13]))
    # print(np.linalg.cond(S))
    # exit()

    # md track parameters 
    LTS = np.ones(64)
    md_aoa_step = 0.02 # rad
    md_tof_step = 0.5e-9 
    md_tof_max = np.ceil(max_d/3e8*1e9)*1e-9 # for dist up to 100m
    md_aoa_range = np.arange(0,np.pi,md_aoa_step)
    md_tof_range = np.arange(0,md_tof_max,md_tof_step)
    ant_sp = sep
    FFT_SZ = nfft
    # precompute aoa matrix, aod matrix, delayed LTS_T
    md_aoa_matrix = [ np.exp(1j*2*np.pi*ant_sp*np.arange(K)*np.cos(aoa_rad)/l1.reshape(-1,1)) for aoa_rad in md_aoa_range ]
    md_aod_matrix = np.exp(1j*2*np.pi*ant_sp*np.arange(K)*np.cos(np.pi/2)/l1.reshape(-1,1))
    md_U = [ np.fft.ifft(LTS*np.exp(-1j*2*np.pi*tof*3e8/l1)) for tof in md_tof_range]

    # R2F2 precomputation
    r2f2_full_exp_dict = r2f2_helper.precomputer(l1, K, sep, max_d, d_step=1, psi_step=1)
    r2f2_predefined_psis = np.arange(-1,1+1e-4,0.005)
    r2f2_S_list = r2f2.get_Sincs(L=K*sep, all_lambs=l1, K=K, N=len(r2f2_predefined_psis), psi_js=r2f2_predefined_psis)

    # t1 = time.time()
    # for i in range(400):
    #     SD = r2f2.get_SD(L=K*sep, K=K, N=2, all_lambs=l1, d_ns=np.array([7,12]), psi_js=test_aoas[[45,87]])
    # t2 = time.time()
    # for i in range(400):
    #     a = test_aoas[[45,87]]
    #     idx = []
    #     for aa in a:
    #         idx.append(bisect.bisect_left(test_aoas, aa))
    #     # r2f2_D_list = r2f2.get_D(N=2, all_lambs=l1, d_ns=np.array([7,12]))
    #     ddd = np.array([7,12])
    #     r2f2_D_list = [np.diag(np.exp(-2j*np.pi*ddd/lambda_i)) for lambda_i in l1]
    #     # SD = np.array([np.matmul(S[:,idx],D) for S,D in zip(r2f2_S_list,r2f2_D_list)]).reshape(len(l1)*K,2)
    #     SD = np.vstack([np.matmul(S[:,idx],D) for S,D in zip(r2f2_S_list,r2f2_D_list)])
    # t3 = time.time()
    # # print(SD.shape)
    # print(t2-t1)
    # print(t3-t2)
    # exit()
    ############################################################
    if not data_from_pickle:
        params_list=[]
        for n_paths in [2,3,4,5,6]:
            for _ in range(50):
                p = rf_common.get_synth_params_sep_aoa_dist(n_paths, min_d, max_d, dist_sep=8, psi_sep=0.2)
                params_list.append(p)
        # params_list = dataset_gen.get_params_multi_proc(num_chans, max_n_paths, max_d, n_cores, min_d, min_n_paths, d_sep, aoa_sep)
        # exit()
    else:
        # params_list = [(np.array([6.12, 16.45]), np.array([0.75, 0.22]), np.array([ 0.0, 0.0]), np.array([0.32, 0.56]))]
        params_list = [(np.array([6.18, 18.43]), np.array([0.75, 0.22]), np.array([ 0.0, 0.0]), np.cos(np.array([60.7, 73.4])/180*np.pi) ) ]
        # params_list = [rf_common.get_synth_params_sep_aoa_dist(4, min_d, max_d, dist_sep=8, psi_sep=0.2)]
    ############################################################
    name = 'model/' + 'cf2.41_d100_dsep8_dstep4_aoasep0_psistep0.10_bw20_K4_2Dconv1.0_hu512_hl3.h5'
    model = keras.models.load_model(name)
    # name = 'cf2.41_d100_dsep8_dstep2_aoasep0_psistep0.10_bw20_K4_2Dconv1.0_hu512_hl3'
    # with open(name+'_model.json') as f:
    #     data = json.load(f)
    # model = keras.models.model_from_json(data)
    # model.load_weights(name+'_weights.h5')

    
    ############################################################
    
    results = collections.defaultdict(list)
    results['noise_pwr'].append(noise_pwr)
    ii, ii_total = 1, len(params_list)
    for params in params_list:
        print(ii, '/', ii_total)
        ii+=1

        results['ground'].append(params)
        snr = np.random.randint(200,250)/10.0
        results['channel_snr'].append(snr)
        ch_l1 = rf_common.get_chans_from_params(params, K, sep, l1)
        y = np.fft.ifft(ch_l1, axis=0)
        noise = np.random.randn(FFT_SZ,K)+1j*np.random.randn(FFT_SZ,K)
        noise = np.sqrt(noise_pwr)*noise/np.abs(noise)
        sig_pwr = np.sum(np.abs(y)**2)/FFT_SZ/K
        rescale = np.sqrt((noise_pwr*10**(snr/10))/sig_pwr)
        y *= rescale
        y += noise
        ch_l1 = np.fft.fft(y, axis=0)
        ch_l2 = rf_common.get_chans_from_params(params, K2, sep, l2)

        results['beamform_snr_ground'].append(beamforming_snr(ch_l2,ch_l2))
        for algo in ['optml2d','r2f2','mdtrack_mobicom19']:
            t1 = time.time()
            if algo == 'optml2d':
                initial_guesses = get_ml_guesses(model, ch_l1, distances, psis, show_plots=False)
                best_sol, reason = r2f2.r2f2_solver_fixed_init(ch_l1, l1, sep, max_d, initial_guesses, r2f2_S_list, r2f2_predefined_psis, max_num_paths=10)
                d_ns, psi_ns = best_sol[0::2], best_sol[1::2]
                param_guesses = r2f2.get_full_params_method3(d_ns, psi_ns, K, sep, l1, ch_l1)
                # print('====================================')
                # print('optml initial guess',initial_guesses)
                # print('optml',param_guesses)
            elif algo == 'r2f2':
                initial_guesses = r2f2_helper.get_initial_guesses(l1, K, sep, ch_l1, r2f2_full_exp_dict, max_d, psi_step=1, dist_step=1, show_plots=False)
                best_sol, reason = r2f2.r2f2_solver(ch_l1, l1, sep, max_d, initial_guesses, r2f2_S_list, r2f2_predefined_psis,max_num_paths=10)
                d_ns, psi_ns = best_sol[0::2], best_sol[1::2]
                param_guesses = r2f2.get_full_params_method3(d_ns, psi_ns, K, sep, l1, ch_l1)
                # print('====================================')
                # print('r2f2 initial guess',initial_guesses)
                # print('r2f2',param_guesses)
            elif algo == 'mdtrack_mobicom19':
                v = core.decompose(y, noise_pwr, rescale, LTS, ant_sp, FFT_SZ, K, l1, md_aoa_step, md_tof_step, md_aoa_range, md_tof_range, md_aoa_matrix, md_aod_matrix, md_U, plot=False, debug=False)
                # v = [AOA (rad), AOD (rad), TOF (s), DOPPLER, ALPHA (channel attenuation)]
                # param_guess = d_ns, a_ns, phi_ns, psi_ns
                param_guesses = (np.real(3e8*v[:,2]), np.abs(v[:,4]), np.angle(v[:,4]), np.real(np.cos(v[:,0])))
                # print('====================================')
                # print('mdtrack_mobicom19',param_guesses)


            ch_l2_guess = rf_common.get_chans_from_params(param_guesses, K2, sep, l2)
            t2 = time.time()
            results['runtime_'+algo].append(round(t2-t1,2))
            results['param_'+algo].append(param_guesses)
            results['beamform_snr_'+algo].append(beamforming_snr(ch_l2,ch_l2_guess))
            results['locerr_'+algo].append(get_loc_error(params,param_guesses))
            # print('ground truth beam power:', results['beamform_snr_ground'][-1])
            # print(algo,'Time:', results['runtime_'+algo][-1], 'beam power:',results['beamform_snr_'+algo][-1],'loc err:',results['locerr_'+algo][-1])          
                # jj=np.argmin(param_guesses[0])
                # yy=param_guesses[0][jj]*np.exp(1j*np.arccos(param_guesses[3][jj]))
                # print(meth+"-beam1:", t, K2, num_paths, len(d_ns), ch_snr, rf_common.beam_eval(ch_l2, ch_l2_guess, 0.001, data_snr), np.abs(yy-xx))
                # print np.round(param_guesses[0][jj],2), np.arccos(param_guesses[3][jj])/np.pi*180
                # print 'localization error (meter):', np.abs(yy-xx)
                # sys.stdout.flush()
    # print(tabulate(results, headers='keys',tablefmt='github'))
    sio.savemat('sim_result.mat', results)