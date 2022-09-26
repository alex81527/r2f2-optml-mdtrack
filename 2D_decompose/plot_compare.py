import os, sys, json
sys.path.append("../")

if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')

from common_modules import rf_common, sparse_op
import numpy as np
import time, keras, sys
from common_modules import r2f2_helper, r2f2
import operator, bisect

import matplotlib.pyplot as pl

def get_ml_guesses(nrows, ncols, model, channel):
    # nrows = len(psis)
    # ncols = len(distances)
    ch = channel.transpose().ravel()
    ch = np.concatenate([np.real(ch), np.imag(ch)])
    # print(ch.shape, ch.reshape(1,-1).shape)
    pred = model.predict(ch.reshape(1,-1))
    pred = pred.reshape(nrows, ncols)
    
    return pred


def get_2D_guess_heatmap(psis, distances, full_exp_dict, h_mat):
    psi_d_mat = np.zeros([len(psis), len(distances)]).astype(np.complex)
    i = 0
    for psi in psis:
        j = 0
        for d in distances:
            key = (d, psi)
            exp_full = full_exp_dict[key]
            val = h_mat*exp_full
            val = np.sum(val)
            psi_d_mat[i,j] = val
            j+=1
        i+=1
    psi_d_mat = np.abs(psi_d_mat)
    psi_d_mat = np.square(psi_d_mat)
    return psi_d_mat

if __name__ == '__main__':
    cf1 = 2.412e9
    cf2 = cf1+30e6
    c = 3e8
    K = 3
    bw = 20e6
    nfft = 64
    min_d, max_d = 0, 80
    d_step = 1
    n_cores = 4
    psi_step = 1.0
    lowest_amp = 0.05
    distances = np.arange(min_d, max_d, d_step)
    psis = np.cos(np.arange(0,180+psi_step,psi_step)*np.pi/180)[::-1] # for bisect it goes from -1 to 1 
    nrows = len(psis)
    ncols = len(distances)
    min_n_paths = 2
    max_n_paths = 4
    d_sep = 15
    K = 3
    l1 = rf_common.get_lambs(cf1, bw, nfft)
    sep = l1[0]/2

    # model_src = "nl5_hu250_K4_2400_30m.hdf5"
    # model_src = "nl5_hu250_K4_2400_sigma_1.2.hdf5"
    # model = keras.models.load_model(model_src)
    name = 'cf2.41_d0_1_80_bw20_K3_2Dconv1.0_1.0_hu10_hl5'
    with open(name+'_model.json') as f:
        data = json.load(f)
    model = keras.models.model_from_json(data)
    model.load_weights(name+'_weights.h5')

    window = 10
    full_exp_dict = r2f2_helper.precomputer(l1, K, sep, max_d, d_step, psi_step)
    for num_paths in [4]:
        max_num_paths = num_paths+2
        params = []
        params.append( (np.array([5.0, 25.0]),np.array([0.7, 0.3]),np.array([0.0, 0.0]),np.array([0.75, 0.2])) )
        params.append( (np.array([5.0, 20.0]),np.array([0.7, 0.3]),np.array([0.0, 0.0]),np.array([0.75, 0.2])) )
        params.append( (np.array([5.0, 15.0]),np.array([0.7, 0.3]),np.array([0.0, 0.0]),np.array([0.75, 0.2])) )
        params.append( (np.array([5.0, 10.0]),np.array([0.7, 0.3]),np.array([0.0, 0.0]),np.array([0.75, 0.2])) )
        for p in params:
            # p = rf_common.get_synth_params(num_paths, min_d, max_d)
            print(p)
            te_Y = sparse_op.get_2d_sparse_target([p], distances, psis, lowest_amp)
            
            data_snr = np.random.randint(100,150)/10.0
            ch_snr = np.random.randint(240,250)/10.0

            ch_l1 = rf_common.get_chans_from_params(p, K, sep, l1)
            ch_l1 = rf_common.add_noise_array(ch_l1, ch_snr)

            pred_ml = get_ml_guesses(nrows, ncols, model, ch_l1)
            p_d_psi = r2f2_helper.get_peaks(pred_ml, distances, psis, show_plots=False)
            # thetas = np.arange(0,180,psi_step)
            # thetas_rad = thetas*np.pi/180
            # psis = np.cos(thetas_rad)
            pred_r2f2 = get_2D_guess_heatmap(psis, distances, full_exp_dict, ch_l1)
            # pred_r2f2 = pred_r2f2[::-1, :]
            
            pl.figure(figsize=(12,6))
            pl.subplot(1,3,1)
            pl.imshow(te_Y[0].todense(), aspect="auto")
            pl.xticks(np.arange(0,ncols,ncols//3), distances[np.arange(0,ncols,ncols//3)])
            pl.yticks(np.arange(0,nrows,nrows//3), np.arccos(psis[np.arange(0,nrows,nrows//3)])/np.pi*180)
            
            pl.title('ground truth')

            pl.subplot(1,3,2)
            pl.imshow(pred_r2f2, aspect="auto")
            pl.xticks(np.arange(0,ncols,ncols//3), distances[np.arange(0,ncols,ncols//3)])
            pl.yticks(np.arange(0,nrows,nrows//3), np.arccos(psis[np.arange(0,nrows,nrows//3)])/np.pi*180)
            # pl.yticks(list(range(0,len(thetas), len(thetas)//5)), list(thetas[::len(thetas)//5]), fontsize="x-large")
            # pl.xticks(list(range(0,len(distances), len(distances)//5)), list(map(int, list(distances[::len(distances)//5]))), fontsize="x-large")
            # pl.ylabel("Angle of arrival, degrees", fontsize="x-large")
            # pl.xlabel("Distance, meters", fontsize="x-large")
            # pl.tight_layout()
            pl.title('r2f2')
            pl.colorbar()
            # pl.show()
            # pl.savefig("r2f2"+str(i)+".png")
            # pl.close()
            
            # pl.figure(figsize=(5,3))
            pl.subplot(1,3,3)
            pl.imshow(pred_ml, aspect="auto")
            pl.xticks(np.arange(0,ncols,ncols//3), distances[np.arange(0,ncols,ncols//3)])
            pl.yticks(np.arange(0,nrows,nrows//3), np.arccos(psis[np.arange(0,nrows,nrows//3)])/np.pi*180)
            # thetas = np.arange(0,180,9)
            # pl.yticks(list(range(0,len(thetas), len(thetas)//5)), list(thetas[::len(thetas)//5]), fontsize="x-large")
            # pl.xticks(list(range(0, 100, 6)), fontsize="x-large")
            # pl.ylabel("Angle of arrival, degrees", fontsize="x-large")
            # pl.xlabel("Distance, meters", fontsize="x-large")
            # pl.tight_layout()
            pl.title('optml2D')
            pl.colorbar()
            pl.show()
            # pl.savefig("nne"+str(i)+".png")
            # pl.close()
            
            # zer = np.zeros([21,101])
            # thet = np.arange(-1,1,0.1)
            # for j in range(num_paths):
            #     d = params[0][j]
            #     aoa = params[-1][j]
            #     xpos = bisect.bisect_left(list(range(31)),d)
            #     ypos = bisect.bisect_left(thet,aoa)
            #     zer[ypos, xpos] = params[1][j]
            # pl.figure(figsize=(5,3))
            # pl.imshow(zer, aspect="auto")
            # thetas = np.arange(0,180,9)
            # pl.yticks(list(range(0,len(thetas), len(thetas)//5)), list(thetas[::len(thetas)//5]), fontsize="x-large")
            # pl.xticks(list(range(0, 100, 6)), fontsize="x-large")
            # pl.ylabel("Angle of arrival, degrees", fontsize="x-large")
            # pl.xlabel("Distance, meters", fontsize="x-large")
            # pl.tight_layout()
            # pl.savefig("ideal"+str(i)+".png")
            # pl.close()
            #pl.show()