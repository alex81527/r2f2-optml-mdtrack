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
    path = 'resolution/d0_0.75_16_aoa0_2.5_51_num10_K='+str(3)+'/'
    with open(path+'pred_ml.pckl', 'rb') as f1, open(path+'pred_r2f2.pckl', 'rb') as f2, open(path+'params_dict.pckl', 'rb') as f3, open(path+'snr.pckl', 'rb') as f4, open(path+'runtime_ml.pckl', 'rb') as f5,open(path+'runtime_r2f2.pckl', 'rb') as f6:
        pred_ml = pickle.load(f1)
        pred_r2f2 = pickle.load(f2)
        params_dict = pickle.load(f3)
        snr = pickle.load(f4)
        runtime_ml = pickle.load(f5)
        runtime_r2f2 = pickle.load(f6)
        # pred_ml = pred_r2f2.copy()
    
    # for k,pa_list in param_dict.items():
    #     print(k, '-->', pa_list)
    #     input('enter..')
    # print(param_dict.keys())
    # exit()
    name = 'cf2.41_d0_1_30_bw20_K3_2Dconv4.0_hu500_hl3_do0.00_l2r0.0000'
    with open(path+'log.txt','w') as f:
        for k,pa_list in params_dict.items():
            print(k)
            f.write(str(k)+'\n')
            for i in range(len(pa_list)):
                # print('ground\t', pa_list[i])
                # print('pred_ml\t', pred_ml[k][i])
                # print('pred_r2\t', pred_r2f2[k][i])
                f.write('ground\t' + str(pa_list[i])+'\n')
                f.write('pred_ml\t'+ str(pred_ml[k][i])+'\n')
                f.write('pred_r2\t'+ str(pred_r2f2[k][i])+'\n')
    
    result_ml = collections.defaultdict(float)
    result_r2f2 = collections.defaultdict(float)
    d_threshold = 1
    aoa_threshold = 5
    for key, val in params_dict.items():
        for par,ml,r2 in zip(val,pred_ml[key],pred_r2f2[key]):
            # take ds,as out from par, see if they were correctly estimated
            ## ML
            idx_set = set()
            cnt=0
            for gd,ga in zip(par[0],par[3]):
                for idx,est_d,est_a in zip(range(len(ml[0])),ml[0],ml[3]):
                    if abs(gd-est_d)<=d_threshold and abs(np.arccos(ga)/np.pi*180-np.arccos(est_a)/np.pi*180)<=aoa_threshold:
                        cnt+=1
                        idx_set.add(idx)
                        break
            if cnt==2 and len(idx_set)==2:
                result_ml[key]+=1
            ## R2F2
            idx_set = set()
            cnt=0
            for gd,ga in zip(par[0],par[3]):
                for idx,est_d,est_a in zip(range(len(r2[0])),r2[0],r2[3]):
                    if abs(gd-est_d)<=d_threshold and abs(np.arccos(ga)/np.pi*180-np.arccos(est_a)/np.pi*180)<=aoa_threshold:
                        cnt+=1
                        idx_set.add(idx)
                        break
            if cnt==2 and len(idx_set)==2:
                result_r2f2[key]+=1 
        
        # print('ml:', result_ml[key], 'r2:', result_r2f2[key])
        # cal. percentage of success
        result_ml[key] /= len(val)
        result_r2f2[key] /= len(val)        
        
    aoas = np.arange(0,51,5) # np.arange(0,181,10)
    ds =  np.arange(0,16,1.5) # np.arange(0,25,3)
    nrows, ncols = len(aoas), len(ds)
    prob_ml = np.zeros((len(aoas),len(ds)))
    prob_r2f2 = np.zeros((len(aoas),len(ds)))
    for i in range(len(aoas)):
        for j in range(len(ds)):
            key = (ds[j],aoas[i])
            prob_ml[i,j] = result_ml[key]
            prob_r2f2[i,j] = result_r2f2[key]
    
    pl.figure()
    pl.subplot(1,2,1)
    pl.imshow(prob_r2f2, aspect='auto')
    pl.xticks(np.arange(0,ncols,ncols//3), ds[np.arange(0,ncols,ncols//3)])
    pl.yticks(np.arange(0,nrows,nrows//3), aoas[np.arange(0,nrows,nrows//3)])
    pl.xlabel('Distance (m)')
    pl.ylabel('AoA (deg)')
    pl.title('R2F2')
    pl.subplot(1,2,2)
    pl.imshow(prob_ml, aspect='auto')
    pl.xticks(np.arange(0,ncols,ncols//3), ds[np.arange(0,ncols,ncols//3)])
    pl.yticks(np.arange(0,nrows,nrows//3), aoas[np.arange(0,nrows,nrows//3)])
    pl.title('optml 2D')
    pl.colorbar()
    # pl.show()
    pl.savefig(path+'result.png')


def main():
    data_from_pickle=False
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

    # params_list = dataset_gen.get_params_multi_proc(n_chans=1000, max_n_paths=3, max_d=30, n_processes=4, min_d=0.5, min_n_paths=1, d_sep=1, aoa_sep=5)
    # with open('aoa_tof/d0_1_30_aoa0_5_180/params_list.pckl', 'wb') as f:
    #     pickle.dump(params_list, f)
    # exit()

    # model_src = "new_nl5_hu400_K4_2400_sigma_1.0_210m_step7.hdf5"
    #model_src = "nl5_hu250_K4_2400_sigma_1.2.hdf5"
    # model = keras.models.load_model(model_src)
    # name = 'cf2.41_d0_3_100_bw20_K3_1Dconv11.0_hu500_hl3_do0.00_l2r0.0000'
    name = 'cf2.41_d0_1_30_bw20_K3_2Dconv4.0_hu500_hl3_do0.00_l2r0.0000'
    with open(name+'_model.json') as f:
        data = json.load(f)
    model = keras.models.model_from_json(data)
    model.load_weights(name+'_weights.h5')
    

    #psi1 , psi2 = np.cos(20/180*np.pi), np.cos(50/180*np.pi)
    #params = (np.array([5.0, 6.6]),np.array([0.7, 0.3]),np.array([0.0, 0.0]),np.array([psi1, psi2]))
    # params = (np.array([0.5]),np.array([0.7]),np.array([0.0]),np.array([0.75]))
    # params_list=[] #[(np.array([5]),np.array([1]),np.array([0.0]),np.array([0.0]))]
    N=0
    params_dict=collections.defaultdict(list)
    aoas = np.arange(0,51,2.5) # np.arange(0,181,20)
    ds = np.arange(0,16,0.75) # np.arange(0,25,6)
    if data_from_pickle:
        with open('resolution/d0_0.75_16_aoa0_2.5_51_num10_K=3/params_dict.pckl', 'rb') as f:
            params_dict = pickle.load(f)
            N = sum([len(v) for v in params_dict.values()])
    else:
        for aoa in aoas:
            for d in ds:
                key = (d,aoa)
                if aoa==0 and d==0:
                    continue
                for _ in range(10):
                    d1 = np.random.rand()*(max_d-d)
                    aoa1 = np.random.rand()*(180-aoa)
                    d_ = np.array([d1   , d1+d])
                    a_ = np.array([1.0   , 0.2+np.random.rand(1)[0]])
                    phi_ = np.array([0.0 , np.random.rand(1)[0]])
                    psi_ = np.array([np.cos(aoa1/180*np.pi) , np.cos((aoa1+aoa)/180*np.pi)])
                    # params_list.append((d_, a_, phi_, psi_))
                    params_dict[key].append((d_, a_, phi_, psi_))
                    N+=1
    print('----------------------------------------------------------')
    # path = 'resolution/d0_1.5_16_aoa0_5_51_num10_K=3/'
    # os.makedirs(path, exist_ok=True)
    # with open(path+'params_dict.pckl', 'wb') as f3:
    #     pickle.dump(params_dict, f3)
    # exit()

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
    for K in [3]:
        pred_ml = collections.defaultdict(list)
        pred_r2f2 = collections.defaultdict(list)
        SNR = collections.defaultdict(list)
        runtime_ml = []
        runtime_r2f2 = []
        full_exp_dict = r2f2_helper.precomputer(l1, K, sep, max_d, d_step, psi_step)
        i=1
        for key,val in params_dict.items():
            for params in val:
                print('-------',str(i)+'/'+str(N)+'-------', key)
                i+=1
                ch_l1 = rf_common.get_chans_from_params(params, K, sep, l1)
                ch_snr = np.random.randint(200,300)/10.0
                ch_l1 = rf_common.add_noise_array(ch_l1, ch_snr)
                # d1,d2,a1,a2 = params_list[i][0][0], params_list[i][0][1], params_list[i][3][0], params_list[i][3][1]
                # key = (d2-d1 , np.arccos(a2)/np.pi*180-np.arccos(a1)/np.pi*180)
                # print(key)
                # ch_l1 = X[i].reshape(K,-1).transpose()
                if K==3:
                    start = time.time()
                    initial_guesses_ml = get_ml_guesses(distances, psis, model, ch_l1)
                    best_sol_ml, reason_ml = r2f2.r2f2_solver_fixed_init(ch_l1, l1, sep, max_d, initial_guesses_ml, max_n_paths+2)
                    end = time.time()
                    d_ns_ml, psi_ns_ml = best_sol_ml[0::2], best_sol_ml[1::2]
                    param_guesses_ml = r2f2.get_full_params_method3(d_ns_ml, psi_ns_ml, K, sep, l1, ch_l1)
                    pred_ml[key].append(param_guesses_ml)
                    runtime_ml.append(end-start)
                
                start = time.time()
                initial_guesses_r2f2 = r2f2_helper.get_initial_guesses(l1, K, sep, ch_l1, full_exp_dict, max_d, psi_step=psi_step, dist_step=d_step, show_plots=False)
                best_sol_r2f2, reason_r2f2 = r2f2.r2f2_solver(ch_l1, l1, sep, max_d, initial_guesses_r2f2, max_n_paths+2)
                end = time.time()
                d_ns_r2f2, psi_ns_r2f2 = best_sol_r2f2[0::2], best_sol_r2f2[1::2]
                param_guesses_r2f2 = r2f2.get_full_params_method3(d_ns_r2f2, psi_ns_r2f2, K, sep, l1, ch_l1)
                pred_r2f2[key].append(param_guesses_r2f2)
                runtime_r2f2.append(end-start)
                
                SNR[key].append(ch_snr)  
    
        path = 'resolution/d0_0.75_16_aoa0_2.5_51_num10_K='+str(K)+'/'
        os.makedirs(path, exist_ok=True)
        with open(path+'pred_ml.pckl', 'wb') as f1, open(path+'pred_r2f2.pckl', 'wb') as f2, open(path+'params_dict.pckl', 'wb') as f3, open(path+'snr.pckl', 'wb') as f4,open(path+'runtime_ml.pckl', 'wb') as f5,open(path+'runtime_r2f2.pckl', 'wb') as f6:
            pickle.dump(pred_ml, f1)
            pickle.dump(pred_r2f2, f2)
            pickle.dump(params_dict, f3)
            pickle.dump(SNR, f4)
            pickle.dump(runtime_ml, f5)
            pickle.dump(runtime_r2f2, f6)

if __name__ == '__main__':
    a = time.time()
    test()
    print('Time (min):', (time.time()-a)/60)