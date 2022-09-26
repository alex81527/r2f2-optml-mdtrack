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


if __name__ == '__main__':
    ####################### Training Parameters #####################
    num_chans = int(1e5)
    num_chans_test = int(1e4)
    min_d, max_d = 0, 100
    d_step = 3
    n_cores = 6
    psi_step = 9.0
    lowest_amp = 0.05
    distances = np.arange(min_d, max_d, d_step)
    psis = np.cos(np.arange(0,181,psi_step)*np.pi/180)[::-1] # for bisect it goes from -1 to 1 
    min_n_paths = 1
    max_n_paths = 3
    d_sep = 7
    K = 3
    ncols = len(distances)
    nrows = len(psis)
    sigma = 1.0
    norm = True
    cf = 2412e6
    bw = 20e6
    nfft = 64
    l1 = rf_common.get_lambs(cf, bw, nfft)
    sep = l1[0] / 2
    
    configs=[] # h_units, num_h_layers, batch_size, num_epochs, activation
    for s in np.arange(1):
        configs.append( (200,5,32,128,'elu'))
    #############################################################
    # a = dataset_gen.get_params_uniform(1000,30,3,15)
    # params_list = dataset_gen.get_params_multi_proc(num_chans, max_n_paths, max_d, n_cores, min_d, min_n_paths, d_sep)
    # te_params = dataset_gen.get_params_multi_proc(num_chans_test, max_n_paths, max_d, n_cores, min_d, min_n_paths, d_sep)
    
    params_list = dataset_gen.get_params_uniform(num_chans,max_d,max_n_paths,d_sep)
    pl.figure()
    tmp = []
    for p in a:
        tmp.extend(list(p[0]))

    pl.hist(tmp)
    pl.show()
    exit()
    # params_list = [(np.array([8.0]),np.array([0.7]),np.array([0.0]),np.array([0.75]))]
    te_params = dataset_gen.get_params_uniform(num_chans_test,max_d,max_n_paths,d_sep)
    tr_X_complex = dataset_gen.get_array_chans_multi_proc(l1, K, sep, params_list, n_cores, norm)
    te_X_complex = dataset_gen.get_array_chans_multi_proc(l1, K, sep, te_params, n_cores, norm)
    # tr_X_complex = rf_common.add_noise_snr_range(tr_X_complex, 20, 30)
    # te_X_complex = rf_common.add_noise_snr_range(te_X_complex, 20, 30)
    tr_X = dataset_gen.to_reals(tr_X_complex)
    te_X = dataset_gen.to_reals(te_X_complex)
    # tr_Y = sparse_op.get_2d_sparse_target(params_list, distances, psis, lowest_amp)
    # te_Y = sparse_op.get_2d_sparse_target(te_params, distances, psis, lowest_amp)
    tr_Y = dataset_gen.get_1d_output(params_list,distances,psis)
    te_Y = dataset_gen.get_1d_output(te_params,distances,psis)
    # print(type(tr_Y[0]), tr_Y[0].shape)
    for cfg in configs:
        h_units, num_h_layers, batch_size, num_epochs, acti = cfg
        print("I/O shapes:", tr_X.shape, tr_Y.shape,te_X.shape, te_Y.shape, 'K=', K)
        # name = str(round(cf/1e9,2))+"_d"+str(min_d)+"_"+str(d_step)+"_"+str(max_d)+"_bw"+str(int(bw/1e6))+'_K'+str(K)
        # name += "_2Dconv"+'{0:.2f}_d{}_{}_{}_bw{}_K{}_2Dconv{0:.1f}_{1:.1f}_hu{}_hl{}'.format(sigma[0], sigma[1])+"_hu"+str(h_units)+"_hl"+str(num_h_layers)
        name = 'cf{0:.2f}_d{1:d}_{2:d}_{3:d}_bw{4:d}_K{5:d}_BinaryCrossEntropy_{6:s}_hu{7:d}_hl{8:d}'.format(
            cf/1e9, min_d,d_step, max_d, int(bw/1e6),K,acti,h_units,num_h_layers)
        print('Model:',name)
        
        model = keras.models.Sequential()
        for i in range(num_h_layers):
            if i == 0: ### first layer needs extra info
                model.add(keras.layers.Dense(units=h_units, activation=acti, input_dim=tr_X.shape[1]))
            elif i==num_h_layers-1:
                model.add(keras.layers.Dense(units=nrows*ncols, activation=acti))
                # the last dimension has to be there, otherwise 'self cycle fanin' error occurs
                # model.add(keras.layers.Reshape((nrows,ncols))) 
            else:
                model.add(keras.layers.Dense(units=h_units, activation=acti))

        early_stop_low_loss = ml_core.EarlyStoppingByLossVal(monitor='val_loss', value=0.001, verbose=1)
        early_stop_no_gain = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50, verbose=1, mode='auto')
        save_best = keras.callbacks.callbacks.ModelCheckpoint(filepath=name+"_weights.h5", verbose=0, save_best_only=True, save_weights_only=True, period=1)
        # callback_list = [early_stop_low_loss, early_stop_no_gain, save_best]

        def loss_lasso(y_true, y_pred):
            return tf.math.add(tf.math.reduce_sum(tf.math.squared_difference(y_true,y_pred)), tf.math.multiply(tf.constant(0.1),tf.math.reduce_sum(tf.math.abs(y_pred))))
        model.compile(optimizer='adam', 
                    loss=loss_lasso, #keras.losses.BinaryCrossentropy(from_logits=True), 
                    metrics=[loss_lasso])

        keras.utils.print_summary(model)
        # o=model.predict(te_X)
        # print(o.shape)
        # exit()
        # num_epochs = 25
        # batch_size = 25
        steps_per_epoch = num_epochs/batch_size
        start_time = time.time()
        history = model.fit(
            x=tr_X, y=tr_Y,
            batch_size=batch_size,
            epochs=num_epochs,
            verbose=1,
            validation_data=(te_X,te_Y),
            # validation_steps=1,
            # validation_split=0.1,
            # callbacks=callback_list
            )
        end_time = time.time()
        print('Epochs: ', num_epochs, 'Batch size:', batch_size, "Training time (min):", (end_time-start_time)/60)
        # Y_pred = model.predict(te_X)
        
        # plotting.plot_2D_decompose_preds(te_Y, label, Y_pred, distances, psis, 'decompose/'+name+'.png')
        plot_history(history, 'loss', 'loss/'+name+'.png')
        ######################### Save Data ###########################
        model.save_weights(name+'_weights.h5')
        with open(name+"_model.json", 'w') as outfile:
            json.dump(model.to_json(), outfile)
        with open(name+"_history.json", 'w') as outfile:
            json.dump(json.dumps(history.history, default=to_serializable), outfile)

        del model 
        # for c in callback_list:
        #     del c 
        keras.backend.clear_session()
        print("------")


        '''
        if not debug:    
            params_list = params_list[-20:]
            targets = targets[-20:]
            
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
                    plotting.plot_2D_decompose_preds(params_list, Y_pred, distances, psis)
                    pl.savefig( str(K)+"_"+str(max_d)+"m.png")
                    pl.close()
                if not debug:
                    for h_units, num_h_layers in [(100,3), (100,10), (50,3), (50,10)]:
                        name = "nl"+str(num_h_layers)+"_hu"+str(h_units)+"_K"+str(K)+"_"+str(int(cf/1e6))+"_sigma_"+str(sigma[0])+"_"+str(max_d)+"m_step"+str(d_step)
                        print(name)
                        model = keras.models.load_model(name+".hdf5")
                        Y_pred = model.predict(X)
                        plotting.plot_2D_decompose_preds(params_list, Y_pred, distances, psis)
                        pl.savefig(name+"_"+str(max_d)+"m.png")
                        pl.close()
                        keras.backend.clear_session()
        '''
