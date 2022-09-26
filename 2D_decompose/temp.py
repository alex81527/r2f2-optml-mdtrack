import os, sys
sys.path.append("../")
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import pylab as pl
import numpy as np
from common_modules import ml_core, rf_common, dataset_gen, sparse_op, plotting
import bisect


def plot_2D_decompose_preds(params_list, Y_pred, distances, psis, name):
    nrows, ncols = len(psis)+1, len(distances)+1
    pl.figure(figsize=(5,4))
    i = 0
    params = params_list[i]
    amp, posx, posy = get_positions(params, distances, psis)
    p = Y_pred[i]
    p = p.reshape(nrows, ncols)
    max_amp = np.max(p)
    for j in range(len(amp)):
        p[posy[j]*-1, posx[j]] = (1+amp[j])*100
    pl.imshow(p, aspect="auto")
 

def get_positions(params, dists, psis):
    posx, posy, amps = [], [], []
    min_d = min(dists)
    max_d = max(dists)
    for i in range(len(params[0])):
        d = params[0][i]
        a = params[1][i]
        psi = params[-1][i]
        x = bisect.bisect_left(dists, d)
        y = bisect.bisect_left(psis, psi)
        if x>=max_d or x<=min_d:
            continue
        posx.append(x)
        posy.append(y)
        amps.append(a)
    return amps, posx, posy



if __name__ == '__main__':
    num_chans = 20
    debug = True

    #### channel parameters
    min_d, max_d = 0, 200
    n_cores = 6
    d_step = 2.0
    psi_step = 0.02
    lowest_amp = 0.0
    distances = np.arange(min_d, max_d, d_step)
    psis = np.arange(-1, 1, psi_step)
    min_n_paths = 6
    max_n_paths = 6
    d_sep = None

    ### RF and antenna parameters
    cf = 650e6
    norm = True
    bw = 10e6
    nfft = 32
    l1 = rf_common.get_lambs(cf, bw, nfft)
    sep = l1[0]/2

    params = rf_common.get_synth_params(2,0, hi=max_d)
    params[1][:] = [0.8,0.2]
    params[0][:] = [133,48]
    params[3][:] = [-0.79,0.7]
    
    params_list = [params]*100
    #params_list = dataset_gen.get_params_multi_proc(num_chans, max_n_paths, max_d, n_cores, min_d, min_n_paths, d_sep)
    targets = sparse_op.get_2d_sparse_target(params_list, distances, psis, lowest_amp)
    ncols = len(distances)
    nrows = len(psis)
    sigma = [3.3, 3.3]
    
    for K  in [4]:
        X = dataset_gen.get_array_chans_multi_proc(l1, K, sep, params_list, n_cores, norm)
        X_c = X+np.random.randn(X.shape[0], X.shape[1])*0.01 + 0.01j*np.random.randn(X.shape[0], X.shape[1])
        X = dataset_gen.to_reals(X_c)
    
        Y_pred = []
        for x, y in sparse_op.nn_batch_generator_2d(X, targets, 1, sigma, nrows, ncols):
            y = y[::-1,::-1]
            y = np.zeros(y.shape)
            Y_pred.append(y)
            if len(Y_pred)>10:
                break
        plot_2D_decompose_preds(params_list, Y_pred, distances, psis, "temp.png")
        pl.xlabel("Distance, meters", fontsize="xx-large")
        pl.ylabel("Angle of Arrival, Degrees", fontsize="xx-large")
        pl.xticks(fontsize="x-large")
        pl.yticks(np.arange(0, len(psis), 25), np.arange(0, 360, 90), fontsize="x-large")
        pl.xticks(np.arange(0, len(distances), 20), np.arange(0, max_d, 40), fontsize="x-large")
        #pl.grid()
        pl.tight_layout()
        pl.savefig("2d_heat1.png")
        pl.show()    
