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
import collections
import pickle
import time

def get_ml_guesses(distances, psis, model, channel):
    nrows, ncols = len(psis), len(distances)
    ch = channel.transpose().ravel()
    ch = np.concatenate([np.real(ch), np.imag(ch)])
    pred = model.predict(ch.reshape(1,-1))
    pred = pred.reshape(nrows, ncols)
    
    p_d_psi = r2f2_helper.get_peaks_ml(pred, distances, psis, show_plots=False)
    # x= sorted([(i,j,pred[i][j]) for j in range(len(pred[0])) for i in range(len(pred))], key=lambda y: y[2], reverse=True)
    # print(pred.shape, pred)
    # print(x[0:10])
    # exit()
    ####sort guesses by "probability"
    p_d_psi = sorted(list(p_d_psi.items()), key=operator.itemgetter(1), reverse=True)
    p_d_psi = [k for k,v in p_d_psi]
    # print('p_d_psi', p_d_psi)
    temp = []
    # print('before', temp)
    list(map(temp.extend, p_d_psi))
    p_d_psi = temp
    # print('after', temp)
    return p_d_psi


def test():
    path='aoa_tof/d0_1_30_aoa30_5_150/'
    with open(path+'pred_ml.pckl', 'rb') as f1, open(path+'pred_r2f2.pckl', 'rb') as f2, open(path+'params_list.pckl', 'rb') as f3:
        pred_ml = pickle.load(f1)
        pred_r2f2 = pickle.load(f2)
        params_list = pickle.load(f3)

    aoa_err_ml = []
    tof_err_ml = []
    aoa_err_r2f2 = []
    tof_err_r2f2 = []
    for i in range(len(params_list)):
        dist_g, aoa_g = params_list[i][0][0], params_list[i][3][0]
        idx = np.argmin(pred_ml[i][0])
        dist_ml, aoa_ml = pred_ml[i][0][idx], pred_ml[i][3][idx]
        idx = np.argmin(pred_r2f2[i][0])
        dist_r2f2, aoa_r2f2 = pred_r2f2[i][0][idx], pred_r2f2[i][3][idx]

        aoa_err_ml.append( abs(np.arccos(aoa_g)/np.pi*180 - np.arccos(aoa_ml)/np.pi*180) )
        tof_err_ml.append( abs(dist_g-dist_ml) )
        aoa_err_r2f2.append( abs(np.arccos(aoa_g)/np.pi*180 - np.arccos(aoa_r2f2)/np.pi*180) )
        tof_err_r2f2.append( abs(dist_g-dist_r2f2) )

        print(i, 'ground',params_list[i][0],params_list[i][3])
        print('pred_ml',pred_ml[i][0],pred_ml[i][3])
        print('pred_r2',pred_r2f2[i][0],pred_r2f2[i][3])

    # print(964, 'ground',params_list[964][0],params_list[964][3])
    # print('pred_ml',pred_ml[964][0],pred_ml[964][3])
    # print('pred_r2',pred_r2f2[964][0],pred_r2f2[964][3])
    # print(tof_err_ml[964], aoa_err_ml[964], tof_err_r2f2[964], aoa_err_r2f2[964])
    
    # plot or save
    bins = np.arange(0,180,2) 
    pl.figure()
    pl.subplot(2,2,1)
    # h4,b4,p1 = pl.hist(aoa_err_ml, bins, histtype="step", cumulative=True, normed=True)
    pl.hist(aoa_err_ml, 180)
    pl.subplot(2,2,2)
    # h4,b4,p1 = pl.hist(aoa_err_r2f2, bins, histtype="step", cumulative=True, normed=True)
    pl.hist(aoa_err_r2f2, 180)
    pl.subplot(2,2,3)
    h4,b4,p1 = pl.hist(tof_err_ml, bins, histtype="step", cumulative=True, normed=True)
    pl.subplot(2,2,4)
    h4,b4,p1 = pl.hist(tof_err_r2f2, bins, histtype="step", cumulative=True, normed=True)
    pl.show()
    # pl.savefig(path+'result.png')
    
def main():
    min_d, max_d = 0, 30
    d_step = 1
    n_cores = 6
    psi_step = 1.0
    distances = np.arange(min_d, max_d, d_step)
    psis = np.cos(np.arange(0,181,psi_step)*np.pi/180)[::-1] # for bisect it goes from -1 to 1 
    min_n_paths = 3
    max_n_paths = 3
    d_sep = 1.5
    lowest_amp = 0.05
    K = 3
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


    path='aoa_tof/d0_1_30_aoa30_5_150/'
    # params_list = dataset_gen.get_params_multi_proc(n_chans=1000, max_n_paths=3, max_d=30, n_processes=4, min_d=0.5, min_n_paths=1, d_sep=1, aoa_sep=5)
    # with open(path+'params_list.pckl', 'wb') as f:
    #     pickle.dump(params_list, f)
    # exit()
    with open(path+'params_list.pckl', 'rb') as f:
        params_list = pickle.load(f)

    name = 'cf2.41_d0_1_30_bw20_K3_2Dconv4.0_hu500_hl3_do0.00_l2r0.0000'
    with open(name+'_model.json') as f:
        data = json.load(f)
    model = keras.models.model_from_json(data)
    model.load_weights(name+'_weights.h5')
    

    full_exp_dict = r2f2_helper.precomputer(l1, K, sep, max_d, d_step, psi_step)
    #psi1 , psi2 = np.cos(20/180*np.pi), np.cos(50/180*np.pi)
    #params = (np.array([5.0, 6.6]),np.array([0.7, 0.3]),np.array([0.0, 0.0]),np.array([psi1, psi2]))
    # params = (np.array([0.5]),np.array([0.7]),np.array([0.0]),np.array([0.75]))
    # params_list=[] #[(np.array([5]),np.array([1]),np.array([0.0]),np.array([0.0]))]
    N=len(params_list)
    
    # params_dict=collections.defaultdict(list)
    # aoas = np.arange(0,91,5) # np.arange(0,181,20)
    # ds = np.arange(0,16,1) # np.arange(0,25,6)
    # for aoa in aoas:
    #     for d in ds:
    #         key = (d,aoa)
    #         if aoa==0 and d==0:
    #             continue
    #         for _ in range(10):
    #             d1 = np.random.rand()*(max_d-d)
    #             aoa1 = np.random.rand()*(180-aoa)
    #             d_ = np.array([d1   , d1+d])
    #             a_ = np.array([1.0   , 0.2+np.random.rand(1)[0]])
    #             phi_ = np.array([0.0 , np.random.rand(1)[0]])
    #             psi_ = np.array([np.cos(aoa1/180*np.pi) , np.cos((aoa1+aoa)/180*np.pi)])
    #             # params_list.append((d_, a_, phi_, psi_))
    #             params_dict[key].append((d_, a_, phi_, psi_))
    #             N+=1
    print('----------------------------------------------------------')
    # for p in params_list:
    #     print(p)
    #     print(p[0][0], p[3][0])
    #     input('..')
    # exit()
    # X = dataset_gen.get_array_chans_multi_proc(l1, K, sep, params_list, n_cores, False)
    # X = rf_common.add_noise_snr_range(X, 20, 30)
    # print(len(params_list))
    # print(X.shape)
    # exit()
    # X = dataset_gen.to_reals(X)
    # Y = model.predict(X)
    
    pred_ml = []
    pred_r2f2 = []
    i=1
    for params in params_list:
        print('-------',str(i)+'/'+str(N)+'-------')
        i+=1
        ch_l1 = rf_common.get_chans_from_params(params, K, sep, l1)
        ch_snr = np.random.randint(200,300)/10.0
        ch_l1 = rf_common.add_noise_array(ch_l1, ch_snr)
            # d1,d2,a1,a2 = params_list[i][0][0], params_list[i][0][1], params_list[i][3][0], params_list[i][3][1]
            # key = (d2-d1 , np.arccos(a2)/np.pi*180-np.arccos(a1)/np.pi*180)
            # print(key)
            # ch_l1 = X[i].reshape(K,-1).transpose()
        initial_guesses_ml = get_ml_guesses(distances, psis, model, ch_l1)
        best_sol_ml, reason_ml = r2f2.r2f2_solver(ch_l1, l1, sep, max_d, initial_guesses_ml, max_n_paths+2)
        d_ns_ml, psi_ns_ml = best_sol_ml[0::2], best_sol_ml[1::2]
        
        initial_guesses_r2f2 = r2f2_helper.get_initial_guesses(l1, K, sep, ch_l1, full_exp_dict, max_d, psi_step=psi_step, dist_step=d_step, show_plots=False)
        best_sol_r2f2, reason_r2f2 = r2f2.r2f2_solver(ch_l1, l1, sep, max_d, initial_guesses_r2f2, max_n_paths+2)
        d_ns_r2f2, psi_ns_r2f2 = best_sol_r2f2[0::2], best_sol_r2f2[1::2]
        
        param_guesses_ml = r2f2.get_full_params_method3(d_ns_ml, psi_ns_ml, K, sep, l1, ch_l1)
        param_guesses_r2f2 = r2f2.get_full_params_method3(d_ns_r2f2, psi_ns_r2f2, K, sep, l1, ch_l1)
        
        pred_ml.append(param_guesses_ml)
        pred_r2f2.append(param_guesses_r2f2)
        # SNR[key].append(ch_snr)  
        
    with open(path+'pred_ml.pckl', 'wb') as f1, open(path+'pred_r2f2.pckl', 'wb') as f2:
        pickle.dump(pred_ml, f1)
        pickle.dump(pred_r2f2, f2)

if __name__ == '__main__':
    a = time.time()
    test()
    print('Time (min):', (time.time()-a)/60)