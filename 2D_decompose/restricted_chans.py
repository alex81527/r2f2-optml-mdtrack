import numpy as np
import random

def get_params2(d_steps, psi_steps, min_n_paths, max_n_paths, num_chans):
    all_params = []
    for i in range(num_chans):
        n_paths = np.random.randint(min_n_paths, max_n_paths+1)
        params = []
        #a = np.random.rand(n_paths) + 0.05
        #a = a/np.max(a)
        a = np.ones(n_paths)
        psi = random.sample(psi_steps, n_paths)
        #n = (0.5-np.random.rand(n_paths))*0.5
        #psi = psi + n
        
        d = random.sample(d_steps, n_paths)
        n = np.random.rand(n_paths)*0.1
        d = d+n
        
        phi = np.random.rand(n_paths)
        params = np.array(d), np.array(a), np.array(phi), np.array(psi)
        all_params.append(params)
    return all_params


def get_params3(max_d, d_window, num_clusters, per_cluster, num_chans):
    all_params = []
    for i in range(num_chans):
        d_list = []
        a_list = []
        phi_list = []
        psi_list = []
        all_ds = np.arange(0, max_d-1, max_d/3)
        all_ds = set(all_ds)
        for n in range(num_clusters):
            a = np.random.rand(per_cluster) + 0.05
            a = a/np.max(a)
            
            psi = np.cos(np.random.rand(per_cluster)*np.pi)
            
            d_main = random.sample(all_ds, 1)[0]
            all_ds.remove(d_main)
            d_cluster = np.arange(d_main, d_main+d_window, 1)
            d = random.sample(d_cluster, per_cluster)
            
            phi = np.random.rand(per_cluster)
            
            d_list.extend(d)
            a_list.extend(a)
            psi_list.extend(psi)
            phi_list.extend(phi)
            
        params = np.array(d_list), np.array(a_list), np.array(phi_list), np.array(psi_list)
        all_params.append(params)
    return all_params