import os, sys, json
sys.path.append("../")

if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')

from common_modules import rf_common, dataset_gen, sparse_op
import numpy as np
import time, keras, sys
from common_modules import r2f2_helper, r2f2
import operator
import pylab as pl
from scipy import signal
from scipy import ndimage

def get_ml_guesses(model, channel, distances,psis,show_plots=True):
    nrows = len(psis)
    ncols = len(distances)
    ch = channel.transpose().ravel()
    ch = np.concatenate([np.real(ch), np.imag(ch)])
    pred = model.predict(ch.reshape(1,-1))
    pred = pred.reshape(nrows, ncols)
    pl.subplot(2, 2, 1)
    pl.imshow(pred, aspect='auto',cmap=pl.get_cmap('jet'))

    # p_d_psi = r2f2_helper.get_peaks(pred, distances, psis, show_plots=True)
    psi_d_mat = pred
    r,c = psi_d_mat.shape
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
    p_d_psi = [k for k, v in p_d_psi]
    temp = []
    list(map(temp.extend, p_d_psi))
    p_d_psi = temp

    return p_d_psi

if __name__ == '__main__':
    min_d, max_d = 0, 100
    n_cores = 6
    d_step = 4
    psi_step = 0.1
    d_sep = 8 # in meters
    aoa_sep = 0 # in degrees
    distances = np.arange(min_d, max_d, d_step)
    psis = np.arange(-1, 1, psi_step) #np.cos(np.arange(0,181,psi_step)*np.pi/180)[::-1] # for bisect it goes from -1 to 1 
    min_n_paths = 1
    max_n_paths = 3
    lowest_amp = 0.05
    K = 4
    ncols = len(distances)
    nrows = len(psis)
    sigma = 1.0
    norm = True
    cf1 = 2412e6
    cf2 = cf1+30e6
    c = 3e8
    bw = 20e6
    nfft = 64
    l1 = rf_common.get_lambs(cf1, bw, nfft)
    l2 = rf_common.get_lambs(cf2, bw, nfft)
    sep = l1[0]/2

    # model_src = "new_nl5_hu400_K4_2400_sigma_1.0_210m_step7.hdf5"
    #model_src = "nl5_hu250_K4_2400_sigma_1.2.hdf5"
    # model = keras.models.load_model(model_src)
    # name = 'cf2.41_d0_3_100_bw20_K3_1Dconv11.0_hu500_hl3_do0.00_l2r0.0000'
    name = 'model/' + 'cf2.41_d100_dsep8_dstep4_aoasep0_psistep0.10_bw20_K4_2Dconv1.0_hu512_hl3.h5'
    model = keras.models.load_model(name)
    # name = 'cf2.41_d100_dsep8_dstep4_aoasep0_psistep0.10_bw20_K4_2Dconv1.0_hu512_hl3'
    # with open(name+'_model.json') as f:
    #     data = json.load(f)
    # model = keras.models.model_from_json(data)
    # model.load_weights(name+'_weights.h5')
    
    def FSPL(dist, freq):
        return 20*np.log10(dist) + 20*np.log10(freq)-147.56
    def db2mag(x):
        return 10**(x/20)
    # p=FSPL(10,2.4e9)
    # print(p,db2mag(-p))
    # exit()
    a1,a2 = 0.7,0.2 #db2mag(-FSPL(8,cf1)), db2mag(-FSPL(17,cf1))
    full_exp_dict = r2f2_helper.precomputer(l1, K, sep, max_d, d_step=1, psi_step=1)
    psi1 , psi2 , psi3 = np.cos(53/180*np.pi), np.cos(83/180*np.pi), np.cos(125/180*np.pi)
    # params = (np.array([10, 14,19]),np.array([0.7, 0.2, 0.1]),np.array([0.0, 0.0, 0.0]),np.array([psi1, psi2, psi3]))
    # params = (np.array([13, 23]),np.array([a1,a2]),np.array([0.0, 0.0]),np.array([psi1, psi2]))
    # params = (np.array([10]),np.array([0.7]),np.array([0.0]),np.array([psi1]))
    params = (np.array([6.18, 18.43]), np.array([0.75, 0.22]), np.array([ 0.0, 0.0]), np.cos(np.array([60.7, 73.4])/180*np.pi) )
    # params = rf_common.get_synth_params_sep(2, min_d, max_d, 8)
    print('----------------------------------------------------------')
    print('params:', params)
    # params = (np.array([8.0]),np.array([0.7]),np.array([0.0]),np.array([0.75]))
    # params = (np.array([5.0, 15.0]),np.array([0.7, 0.3]),np.array([0.0, 0.0]),np.array([0.75, 0.2])) 
    ch_snr = np.random.randint(200,250)/10.0
    ch_l1 = rf_common.get_chans_from_params(params, K, sep, l1)
    ch_l1 = rf_common.add_noise_array(ch_l1, ch_snr)
    '''
    params2 = (np.array([11.23850094]), np.array([1.]), np.array([-0.58036774]), np.array([0.40066825]))
    # params2 = (np.array([12.7217758 , 17.68712288]), np.array([0.44886826, 0.55113174]), np.array([ 0.76747872, -0.44228476]), np.array([0.32431135, 0.05431461]))
    ch_l2 = rf_common.get_chans_from_params(params2, K, sep, l1)
    ch_snr = 20.0
    p1 = np.mean( np.abs( np.sum(ch_l1,axis=1 ) )**2)
    noise_pwr = p1 / (10**(ch_snr/10))
    p2 = np.mean( np.abs( np.sum(ch_l1/(ch_l2/np.abs(ch_l2)),axis=1 ) )**2)
    p3 = np.mean( np.abs( np.sum(ch_l1/(ch_l1/np.abs(ch_l1)),axis=1 ) )**2)
    print(10*np.log10(p1/noise_pwr))
    print(10*np.log10(p2/noise_pwr))
    print('snr gain:', 10*np.log10(p2/p1))
    print('snr gain:', 10*np.log10(p3/p1))
    print(p2,p3)
    # print('snr (no beam)', 10*np.log10(p1/noise_pwr))
    # print('snr (beamforming)', 10*np.log10(p2/noise_pwr))
    # print('SNR gain', 10*np.log10(p2/p1))
    # pl.figure()
    # pl.plot(abs( (ch_l1/np.exp(np.angle(ch_l2))).transpose().ravel()))
    # pl.show()
    exit()
    '''

    
    te_X = dataset_gen.to_reals(ch_l1.transpose().ravel())
    te_Y = sparse_op.get_2d_sparse_target([params], distances, psis, lowest_amp)
    temp = ndimage.gaussian_filter(te_Y[0].todense(), 1.0)
    norm = np.max(temp)
    norm = 1 if norm ==0 else norm
    temp = 100*np.log2(((temp/norm)**0.5)+1)
    te_Y = temp
    # print(te_X.shape, te_Y.shape)
    yyy = model.predict(te_X.reshape(1,nfft*K*2))
    
    # pl.figure()
    # pl.imshow(te_Y)
    # pl.xticks(np.arange(0,ncols,ncols//3), distances[np.arange(0,ncols,ncols//3)])
    # pl.yticks(np.arange(0,nrows,nrows//3), np.arccos(psis[np.arange(0,nrows,nrows//3)])/np.pi*180)
    # pl.colorbar()
    # pl.title('te_Y')
    # pl.show()
    
    
    pl.figure()
    initial_guesses_r2f2 = r2f2_helper.get_initial_guesses(l1, K, sep, ch_l1, full_exp_dict, max_d, psi_step=1, dist_step=1, show_plots=True)
    print('r2f2 initial guess', initial_guesses_r2f2)
    pl.figure()
    ml_guess = get_ml_guesses(model, ch_l1,distances, psis)
    print('optml initial guess', ml_guess)
    

    # pl.figure()
    # p_d_psi = r2f2_helper.get_peaks_ml(yyy.reshape(nrows,ncols), distances, psis, show_plots=True)
    # print(p_d_psi)
    # p_d_psi = sorted(list(p_d_psi.items()), key=operator.itemgetter(1), reverse=True)
    # p_d_psi = [k for k,v in p_d_psi]
    # temp = []
    # list(map(temp.extend, p_d_psi))
    # p_d_psi = temp
    # # initial_guesses_ml = p_d_psi
    # d_psis = r2f2_helper.fix_conditioning(p_d_psi, 30, 2, 15)
    # print('after conditioning:', d_psis)
    pl.show()
    exit()
    

    # pl.subplot(1,2,1)
    # pl.imshow(yyy[0].reshape(nrows,ncols))
    # pl.colorbar()
    # pl.title('optml')
    # pl.subplot(1,2,2)
    # psi_d_mat = r2f2_helper.get_2D_guess_heatmap(psis, distances, full_exp_dict, ch_l1)
    # pl.imshow(psi_d_mat)
    # pl.colorbar()
    # pl.title('r2f2')
    # q,r=signal.deconvolve(yyy[0],signal.gaussian(50,11))
    # pl.plot(yyy[0])
    # pl.imshow(yyy[0].reshape(nrows,ncols))
    # pl.colorbar()
    # pl.show()
    # exit()

    
    initial_guesses_ml = get_ml_guesses(distances, psis, model, ch_l1)
    best_sol_ml, reason_ml = r2f2.r2f2_solver_fixed_init(ch_l1, l1, sep, max_d, initial_guesses_ml, max_n_paths+2)
    print(reason_ml)
    d_ns_ml, psi_ns_ml = best_sol_ml[0::2], best_sol_ml[1::2]
    param_guesses_ml = r2f2.get_full_params_method3(d_ns_ml, psi_ns_ml, K, sep, l1, ch_l1)
    
    initial_guesses_r2f2 = r2f2_helper.get_initial_guesses(l1, K, sep, ch_l1, full_exp_dict, max_d, psi_step=psi_step, dist_step=d_step, show_plots=False)
    # initial_guesses_r2f2 = [10, 0.484, 10+2*np.random.rand(1)[0] ,0.484+0.1*np.random.rand(1)[0],10+2*np.random.rand(1)[0] ,0.484+0.1*np.random.rand(1)[0],10+2*np.random.rand(1)[0] ,0.484+0.1*np.random.rand(1)[0],10+2*np.random.rand(1)[0] ,0.484+0.1*np.random.rand(1)[0]]
    best_sol_r2f2, reason_r2f2 = r2f2.r2f2_solver(ch_l1, l1, sep, max_d, initial_guesses_r2f2, max_n_paths+2)
    print(reason_r2f2)
    d_ns_r2f2, psi_ns_r2f2 = best_sol_r2f2[0::2], best_sol_r2f2[1::2]
    param_guesses_r2f2 = r2f2.get_full_params_method3(d_ns_r2f2, psi_ns_r2f2, K, sep, l1, ch_l1)

    print('----------------------------------------------------------')
    print('2Doptml\n','init_guess ', initial_guesses_ml, '\nfinal ', param_guesses_ml)
    print('----------------------------------------------------------')
    print('R2F2\n','init_guess ', initial_guesses_r2f2,'\nfinal ', param_guesses_r2f2)
        
    ch_ml = rf_common.get_chans_from_params(param_guesses_ml, K, sep, l1)
    ch_r2f2 = rf_common.get_chans_from_params(param_guesses_r2f2, K, sep, l1)
    
    param_guesses_r2f2 = (np.array([10]),np.array([0.7]),np.array([0.0]),np.array([np.cos(70/180*np.pi)]))
    ch_ml_l2 = rf_common.get_chans_from_params(param_guesses_ml, K, sep, l2)
    ch_r2f2_l2 = rf_common.get_chans_from_params(param_guesses_r2f2, K, sep, l2)
    ch_l2 = rf_common.get_chans_from_params(params, K, sep, l2)
    
    p1 = np.mean( np.abs( np.sum(ch_l2,axis=1 ) )**2)
    p2 = np.mean( np.abs( np.sum(ch_l2/(ch_l2/np.abs(ch_l2)),axis=1 ) )**2)
    p3 = np.mean( np.abs( np.sum(ch_l2/(ch_ml_l2/np.abs(ch_ml_l2)),axis=1 ) )**2)
    p4 = np.mean( np.abs( np.sum(ch_l2/(ch_r2f2_l2/np.abs(ch_r2f2_l2)),axis=1 ) )**2)
    print('optimal snr gain (dB) :', 10*np.log10(p2/p1))
    print('optml snr gain (dB) :', 10*np.log10(p3/p1))
    print('R2F2 snr gain (dB) :', 10*np.log10(p4/p1))
    exit()
    '''
    pl.figure()
    pl.subplot(2,1,1)
    pl.plot(np.abs(ch_l1.transpose().ravel()))
    pl.plot(np.abs(ch_ml.transpose().ravel()))
    pl.plot(np.abs(ch_r2f2.transpose().ravel()))
    pl.subplot(2,1,2)
    pl.plot(np.angle(ch_l1.transpose().ravel()))
    pl.plot(np.angle(ch_ml.transpose().ravel()))
    pl.plot(np.angle(ch_r2f2.transpose().ravel()))
    pl.legend(['ground truth', 'optml', 'r2f2'])    
    pl.title('l1')

    pl.figure()
    pl.subplot(2,1,1)
    pl.plot(20*np.log10(np.abs(ch_l2.transpose().ravel())))
    pl.plot(20*np.log10(np.abs(ch_ml_l2.transpose().ravel())))
    pl.plot(20*np.log10(np.abs(ch_r2f2_l2.transpose().ravel())))
    pl.subplot(2,1,2)
    pl.plot(np.angle(ch_l2.transpose().ravel()))
    pl.plot(np.angle(ch_ml_l2.transpose().ravel()))
    pl.plot(np.angle(ch_r2f2_l2.transpose().ravel()))
    pl.legend(['ground truth', 'optml', 'r2f2'])
    pl.title('l2')
    pl.show()
    

    print('SNR', ch_snr)
    pl.figure(figsize=(15,8))
    pl.subplot(2,2,1)
    r2f2_2dmap = r2f2_helper.get_2D_guess_heatmap(psis, distances, full_exp_dict, ch_l1)
    pl.imshow(r2f2_2dmap, aspect='auto')
    for d,a in zip(params[0],np.arccos(params[3])/np.pi*180):
        pl.plot(d/d_step, len(psis)-a/psi_step, 'k^',markersize=6, label='ground truth')
    for d,a in zip(param_guesses_r2f2[0],np.arccos(param_guesses_r2f2[3])/np.pi*180):
        pl.plot(d/d_step, len(psis)-a/psi_step, 'bx',markersize=6, label='R2F2')
    for d,a in zip(param_guesses_ml[0],np.arccos(param_guesses_ml[3])/np.pi*180):
        pl.plot(d/d_step, len(psis)-a/psi_step, 'r*',markersize=6, label='ML')
    pl.xticks(np.arange(0,ncols,ncols//3), distances[np.arange(0,ncols,ncols//3)])
    pl.yticks(np.arange(0,nrows,nrows//3), np.arccos(psis[np.arange(0,nrows,nrows//3)])/np.pi*180)
    pl.legend()
    pl.colorbar()

    pl.subplot(2,2,2)
    pl.imshow(te_Y,aspect='auto')
    # pl.contour(distances, psis, te_Y)
    for d,a in zip(params[0],np.arccos(params[3])/np.pi*180):
        pl.plot(d/d_step, len(psis)-a/psi_step, 'k^',markersize=6, label='ground truth')
    pl.xticks(np.arange(0,ncols,ncols//3), distances[np.arange(0,ncols,ncols//3)])
    pl.yticks(np.arange(0,nrows,nrows//3), np.arccos(psis[np.arange(0,nrows,nrows//3)])/np.pi*180)
    pl.colorbar()
    pl.legend()
    pl.title('te_Y')

    pl.subplot(2,2,3)
    pl.imshow(yyy.reshape(nrows,ncols),aspect='auto') 
    # tmp = ch_l1 - rf_common.get_chans_from_params(param_guesses_r2f2, K, sep, l1)
    # print('p1', param_guesses_r2f2)
    # initial_guesses_r2f2 = r2f2_helper.get_initial_guesses(l1, K, sep, tmp, full_exp_dict, max_d, psi_step=psi_step, dist_step=d_step, show_plots=False)
    # best_sol_r2f2, reason_r2f2 = r2f2.r2f2_solver(tmp, l1, sep, max_d, initial_guesses_r2f2, max_n_paths+2)
    # print(reason_r2f2)
    # d_ns_r2f2, psi_ns_r2f2 = best_sol_r2f2[0::2], best_sol_r2f2[1::2]
    # param_guesses_r2f2 = r2f2.get_full_params_method3(d_ns_r2f2, psi_ns_r2f2, K, sep, l1, ch_l1)
    # print('p2', param_guesses_r2f2)
    # tmp2 = tmp - rf_common.get_chans_from_params(param_guesses_r2f2, K, sep, l1)
    # new_2dmap = r2f2_helper.get_2D_guess_heatmap(psis, distances, full_exp_dict, tmp2)
    # pl.imshow(new_2dmap, aspect='auto')
    # pl.colorbar()
    
    pl.subplot(2,2,4)
    # tmp = ch_l1 - rf_common.get_chans_from_params(param_guesses_ml, K, sep, l1)
    # new_2dmap = r2f2_helper.get_2D_guess_heatmap(psis, distances, full_exp_dict, tmp)
    # pl.imshow(new_2dmap, aspect='auto')
    pl.contour(distances, psis, r2f2_2dmap)
    pl.show()
    '''
    exit()
