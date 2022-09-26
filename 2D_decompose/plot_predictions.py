import os, sys
sys.path.append("../")
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import pylab as pl
import numpy as np
from common_modules import ml_core, rf_common, dataset_gen, sparse_op, plotting
import keras
import copy
from keras.optimizers import Adam, RMSprop, SGD
from . import restricted_chans


def heat_map(psis, distances, h, sep, K, lambs):
    op = np.zeros([len(distances), len(psis)]).astype(np.complex)
    for i in range(len(distances)):
        for j in range(len(psis)):
            for k in range(K):
                d = distances[i]
                p = psis[j]
                v = np.exp(2j*np.pi*(d+ k*sep*p)/lambs)
                op[i,j] += np.sum(h[:,k]*v)
    op = np.abs(op)**2
    op = op.transpose()
    th = np.percentile(op, 90)
    op += th
    op = np.log(op)
    return op


if __name__ == '__main__':


    debug = False
    if len(sys.argv) == 2:
        num_chans = 20
        debug = True
    else:
        num_chans = 150
        
    #### channel parameters
    min_d, max_d = 0, 30
    d_step = 1

    for max_d, d_step in [(210,7)]:
        n_cores = 6
        psi_step = 0.1
        lowest_amp = 0.05
        distances = np.arange(min_d, max_d, d_step)
        psis = np.arange(-1, 1, psi_step)
        min_n_paths = 6
        max_n_paths = 6
        d_sep = 15
        K = 4
    
        params_list = dataset_gen.get_params_multi_proc(num_chans, max_n_paths, max_d, n_cores, min_d, min_n_paths, d_sep)
    
        targets = sparse_op.get_2d_sparse_target(params_list, distances, psis, lowest_amp)
        ncols = len(distances)
        nrows = len(psis)
        sigma = [0.90, 0.90]
        #sigma = [1.0, 1.0]
        
        for cf in [2.4e9]:
            norm = False
            bw = 10e6
            nfft = 64
            l1 = rf_common.get_lambs(cf, bw, nfft)
            sep = l1[0]/2
            
            X = dataset_gen.get_array_chans_multi_proc(l1, K, sep, params_list, n_cores, norm)
            print("K=", K," Xshape:",X.shape)
            X = X+np.random.randn(X.shape[0], X.shape[1])*0.005 + 0.005j*np.random.randn(X.shape[0], X.shape[1])
            X = dataset_gen.to_reals(X)
        
            Y_pred = []
            for x, y in sparse_op.nn_batch_generator_2d(X, targets, 1, sigma, nrows, ncols):
                Y_pred.append(y)
                if len(Y_pred)>20:
                    break
            if K == 4:

                params = params_list[0]
                amp, posx, posy = plotting.get_positions(params, distances, psis)
                p = np.zeros([len(psis)+1, len(distances)+1])
                for j in range(len(amp)):
                    p[posy[j], posx[j]] = amp[j]
                pl.imshow(p)
                pl.xticks(list(range(len(distances)))[::5], np.round(distances,0)[::5], fontsize="x-large")
                pl.yticks(list(range(len(psis)))[::2], np.round(psis,1)[::2], fontsize="x-large")
                pl.ylabel("cos(AoA)", fontsize="x-large")
                pl.xlabel("Distance or delay", fontsize="x-large")
                pl.tight_layout()

                pl.savefig(str(max_d)+"_ideal.png")
                pl.close()
            
            if not debug:
                for h_units in [50, 100, 400, 200]:
                    if h_units in [50, 100]:
                        sigma=[1.0, 1.0]
                    else:
                        sigma=[0.9, 0.9]
                    for num_h_layers in [3, 10]:
                        name = "nl"+str(num_h_layers)+"_hu"+str(h_units)+"_K"+str(K)+"_"+str(int(cf/1e6))+"_sigma_"+str(sigma[0])+"_"+str(max_d)+"m_step"+str(d_step)
                        print(name)
                        try:
                            model = keras.models.load_model(name+".hdf5")
                        except:
                            print("missing")
                            continue
                        Y_pred = model.predict(X)
                        print("found")
            
                        params = params_list[0]
                        p = Y_pred[0]
                        p = p.reshape(len(psis)+1, len(distances)+1)
                        pl.imshow(p)
            
                        pl.xticks(list(range(len(distances)))[::5], np.round(distances,0)[::5], fontsize="x-large")
                        pl.yticks(list(range(len(psis)))[::2], np.round(psis,1)[::2], fontsize="x-large")
                        pl.ylabel("cos(AoA)", fontsize="x-large")
                        pl.xlabel("Distance or delay", fontsize="x-large")
                        pl.tight_layout()
                        pl.savefig(name+"_"+str(max_d)+"m.png")
                        pl.close()
                        keras.backend.clear_session()
            #            



            distances = np.arange(min_d, max_d, 1)
            psis = np.arange(-1, 1, 0.015)
            
            #x = rf_common.get_chans_from_params(params_list[0], K, sep, l1)
            #p = heat_map(psis, distances, x, sep, K, l1)
            #pl.imshow(p)
            #
            #pl.xticks(range(len(distances))[::25], np.round(distances,0)[::25], fontsize="x-large")
            #pl.yticks(range(len(psis))[::20], np.round(psis,1)[::20], fontsize="x-large")
            #pl.ylabel("cos(AoA)", fontsize="x-large")
            #pl.xlabel("Distance or delay", fontsize="x-large")
            #pl.tight_layout()
            #pl.savefig("r2f2.png")
            #pl.close()