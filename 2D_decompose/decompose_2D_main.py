import os
import sys
sys.path.append("../")
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import pylab as pl
import numpy as np
from common_modules import ml_core, rf_common, dataset_gen, sparse_op, plotting
import keras
import tensorflow as tf
import copy
from keras.optimizers import Adam, RMSprop, SGD
import restricted_chans
import json
from functools import singledispatch
import time
import pickle

@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)

@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)

def plot_history(hist, func, filename):
    pl.figure(figsize=(10,8))
    pl.plot(hist.history[func])
    pl.plot(hist.history['val_'+func])
    pl.legend([func, 'val_'+func])
    if filename != "":
        pl.savefig(filename)
        pl.close()
    else:
        pl.show()

def fit_pred_fcnn(X, targets, params_list, psis, distances, sigma, h_units, num_h_layers, name="", fname=""):
    in_dim = X.shape[1]
    callback_list = ml_core.get_callbacks(0.01, name, 50, 0.02)

    nrows = len(psis)
    ncols = len(distances)
    out_dim = (ncols + 1) * (nrows + 1)

    activation = "elu"
    optimizer = "adam"
    loss_fn = "mae"
    batch_size = 256
    num_epochs = 550
    num_chans = X.shape[0]
    steps_per_epoch = num_chans / batch_size
    te_index = int(0.9 * X.shape[0])

    model = ml_core.get_model(in_dim, (nrows + 1, ncols + 1), h_units, num_h_layers, activation)
    model.compile(optimizer, loss_fn, metrics=[loss_fn])
    keras.utils.print_summary(model)
    history = model.fit_generator(
        sparse_op.nn_batch_generator_2d(
            X[:te_index], targets[:te_index], batch_size, sigma, nrows, ncols),
        steps_per_epoch,
        epochs=num_epochs,
        verbose=2,
        validation_data=sparse_op.nn_batch_generator_2d(
            X[te_index:], targets[te_index:], batch_size, sigma, nrows, ncols),
        validation_steps=1,
        # validation_split=0.1,
        callbacks=callback_list
        )
    return model


if __name__ == '__main__':
    ########################## Parameters #####################
    # ML parameter
    data_from_pickle = True
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
    cf = 2412e6
    bw = 20e6
    nfft = 64
    l1 = rf_common.get_lambs(cf, bw, nfft) # wavelengths
    sep = l1[0] / 2
    ################## Data Generation ####################################
    if not data_from_pickle:
        folder = 'training_data/cf{0:.2f}_d{1:d}_dsep{2:d}_aoasep{3:d}_bw{4:d}_K{5:d}_mp{6:d}_num{7:.0e}/'.format(
            cf/1e9, max_d,d_sep, aoa_sep, int(bw/1e6), K, max_n_paths, num_chans)
        print('save data to '+folder+'...')
        print('training:',num_chans, 'test:',num_chans_test)
        params_list = dataset_gen.get_params_multi_proc(num_chans, max_n_paths, max_d, n_cores, min_d, min_n_paths, d_sep, aoa_sep)
        # ds_ = []
        # aoas_ = []
        # for p in params_list:
        #     ds_.extend(list(p[0]))
        #     aoas_.extend(list(p[3]))
        # pl.figure()
        # pl.hist(ds_)
        # pl.figure()
        # pl.hist(np.arccos(aoas_)/np.pi*180)
        # pl.show()
        te_params = dataset_gen.get_params_multi_proc(num_chans_test, max_n_paths, max_d, n_cores, min_d, min_n_paths, d_sep, aoa_sep)
        tr_X_complex = dataset_gen.get_array_chans_multi_proc(l1, K, sep, params_list, n_cores, norm)
        te_X_complex = dataset_gen.get_array_chans_multi_proc(l1, K, sep, te_params, n_cores, norm)
        tr_X_complex = rf_common.add_noise_snr_range(tr_X_complex, 20, 30)
        te_X_complex = rf_common.add_noise_snr_range(te_X_complex, 20, 30)
        tr_X = dataset_gen.to_reals(tr_X_complex)
        te_X = dataset_gen.to_reals(te_X_complex)
        
        tr_Y = sparse_op.get_2d_sparse_target(params_list, distances, psis, lowest_amp)
        te_Y = sparse_op.get_2d_sparse_target(te_params, distances, psis, lowest_amp)
        '''
        tr_Y = dataset_gen.get_1d_output(params_list, distances, psis) # amplify target matrix by 50
        te_Y = dataset_gen.get_1d_output(te_params, distances, psis)
        '''
        # print(te_Y.shape, te_Y[0].shape, '\n-----')
        os.makedirs(folder, exist_ok=True)
        with open(folder+'params_list.pckl', 'wb') as f:
            pickle.dump(params_list, f)
        with open(folder+'te_params.pckl', 'wb') as f:
            pickle.dump(te_params, f)
        with open(folder+'tr_X.pckl', 'wb') as f:
            pickle.dump(tr_X, f)
        with open(folder+'tr_Y.pckl', 'wb') as f:
            pickle.dump(tr_Y, f)
        with open(folder+'te_X.pckl', 'wb') as f:
            pickle.dump(te_X, f)
        with open(folder+'te_Y.pckl', 'wb') as f:
            pickle.dump(te_Y, f)
        exit()
    #############################################################
    configs=[] # h_units, num_h_layers, batch_size, num_epochs, sigma
    for s in [1.0]:
        for hu in [nfft*K*2]:
            for hl in [3]:
                for dropout in [0.0]:
                    for l2reg in [0.0]:
                        for d_step in [4,2]:
                            for psi_step in [0.1,0.05]:
                                for aoa_sep in [0]: # rad
                                    configs.append( (aoa_sep, d_step,psi_step,hu,hl,64,1024*3,s,dropout,l2reg))
    #############################################################
    for cfg in configs:
        aoa_sep,d_step,psi_step,h_units, num_h_layers, batch_size, num_epochs, sigma, dropout, l2reg = cfg
        folder = 'training_data/cf{0:.2f}_d{1:d}_dsep{2:d}_aoasep{3:d}_bw{4:d}_K{5:d}_mp{6:d}_num{7:.0e}/'.format(
            cf/1e9, max_d,d_sep, aoa_sep, int(bw/1e6), K, max_n_paths, num_chans)
        with open(folder+'params_list.pckl', 'rb') as f:
            params_list = pickle.load(f)
        with open(folder+'te_params.pckl', 'rb') as f:
            te_params = pickle.load(f)
        with open(folder+'tr_X.pckl', 'rb') as f:
            tr_X = pickle.load(f)
        with open(folder+'tr_Y.pckl', 'rb') as f:
            tr_Y = pickle.load(f)
        with open(folder+'te_X.pckl', 'rb') as f:
            te_X = pickle.load(f)
        with open(folder+'te_Y.pckl', 'rb') as f:
            te_Y = pickle.load(f)
        distances = np.arange(min_d, max_d, d_step)
        psis = np.arange(-1, 1, psi_step)
        ncols = len(distances)
        nrows = len(psis) 
        tr_Y = sparse_op.get_2d_sparse_target(params_list, distances, psis, lowest_amp)
        te_Y = sparse_op.get_2d_sparse_target(te_params, distances, psis, lowest_amp)
        # sigma = (sigma1,sigma2)
        print("I/O shapes:", tr_X.shape, tr_Y.shape,te_X.shape, te_Y.shape, 'K=', K)
        # name = 'cf{0:.2f}_d{1:d}_{2:d}_{3:d}_bw{4:d}_K{5:d}_2Dconv{6:.1f}_{7:.1f}_hu{8:d}_hl{9:d}_do{10:.2f}_l2r{11:.4f}'.format(
        #     cf/1e9, min_d,d_step, max_d, int(bw/1e6),K, sigma1,sigma2,h_units,num_h_layers,dropout,l2reg)
        name = 'cf{0:.2f}_d{1:d}_dsep{2:d}_dstep{3:d}_aoasep{4:d}_psistep{5:.2f}_bw{6:d}_K{7:d}_2Dconv{8:.1f}_hu{9:d}_hl{10:d}'.format(
            cf/1e9, max_d,d_sep, d_step, aoa_sep, psi_step,int(bw/1e6),K, sigma, h_units,num_h_layers)
        # name = 'cf{0:.2f}_d{1:d}_{2:d}_{3:d}_bw{4:d}_K{5:d}_lasso{6:.1f}_hu{7:d}_hl{8:d}_do{9:.2f}_l2r{10:.4f}'.format(
        #     cf/1e9, min_d,d_step, max_d, int(bw/1e6),K, sigma ,h_units,num_h_layers,dropout,l2reg)
        print('Model:',name)
        
        model = keras.models.Sequential()
        for i in range(num_h_layers):
            if i == 0: ### first layer needs extra info
                model.add(keras.layers.Dense(units=h_units, activation='elu', input_dim=tr_X.shape[1],
                            kernel_regularizer=keras.regularizers.l2(l2reg) if l2reg>0 else None))
            elif i==num_h_layers-1:
                model.add(keras.layers.Dense(units=nrows*ncols, activation='elu'))
                # the last dimension has to be there, otherwise 'self cycle fanin' error occurs
                model.add(keras.layers.Reshape((nrows,ncols,1))) 
            else:
                model.add(keras.layers.Dense(units=h_units, activation='elu',
                            kernel_regularizer=keras.regularizers.l2(l2reg) if l2reg>0 else None))
                if dropout>0.0:
                    model.add(keras.layers.Dropout(dropout))

        early_stop_low_loss = ml_core.EarlyStoppingByLossVal(monitor='val_loss', value=0.001, verbose=1)
        early_stop_no_gain = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='auto')
        os.makedirs('model', exist_ok=True)
        save_best = keras.callbacks.callbacks.ModelCheckpoint(filepath='model/'+name+'.h5', verbose=0, save_best_only=True, save_weights_only=False, period=1)
        callback_list = [save_best] #[early_stop_low_loss, early_stop_no_gain, save_best]

        def loss_lasso(y_true, y_pred):
            return tf.math.add(tf.math.reduce_sum(tf.math.squared_difference(y_true,y_pred)), tf.math.multiply(tf.constant(sigma),tf.math.reduce_sum(tf.math.abs(y_pred))))
        model.compile(optimizer='adam', loss='mae', metrics=['mae'])

        # keras.utils.print_summary(model)
        # o=model.predict(te_X)
        # print(o.shape)
        # exit()
        # num_epochs = 25
        # batch_size = 25
        steps_per_epoch = num_epochs/batch_size
        start_time = time.time()
        
        history = model.fit_generator(
            sparse_op.nn_batch_generator_2d(tr_X, tr_Y, batch_size, sigma, nrows, ncols),
            steps_per_epoch,
            epochs=num_epochs,
            verbose=0,
            validation_data=sparse_op.nn_batch_generator_2d(te_X, te_Y, batch_size, sigma, nrows, ncols),
            validation_steps=1,
            # validation_split=0.1,
            callbacks=callback_list
            )
        '''
        history = model.fit(
            x=tr_X, y=tr_Y,
            epochs=num_epochs,
            verbose=0,
            validation_data=(te_X, te_Y), 
            callbacks=callback_list
            )
        '''
        end_time = time.time()
        print('Epochs: ', num_epochs, "Training time (min):", (end_time-start_time)/60)
        Y_pred = model.predict(te_X)
        # print(Y_pred.shape)
        
        label=[]
        for x, y in sparse_op.nn_batch_generator_2d(te_X, te_Y, 1, sigma, nrows, ncols):
            label.append(y)
            if len(label)>20:
                break
        os.makedirs('decompose', exist_ok=True)
        plotting.plot_2D_decompose_preds(te_Y, label, Y_pred, distances, psis, 'decompose/'+name+'.png')
        '''
        plotting.plot_lasso_pred(te_Y, Y_pred, distances, psis, 'decompose/'+name+'.png')
        '''
        os.makedirs('loss', exist_ok=True)
        plot_history(history, 'loss', 'loss/'+name+'.png')
        ######################### Save Data ###########################
        # with open(name+"_model.json", 'w') as outfile:
        #     json.dump(model.to_json(), outfile)
        # with open(name+"_history.json", 'w') as outfile:
        #     json.dump(json.dumps(history.history, default=to_serializable), outfile)

        del model 
        for c in callback_list:
            del c 
        keras.backend.clear_session()
        print("------")