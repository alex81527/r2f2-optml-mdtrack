import os, sys
sys.path.append("../")
from common_modules import rf_common
import numpy as np
import pylab as pl

if __name__ == '__main__':
	K = 4
	for K in [2, 1]:
		nfft = 32
		sep = 0.2
		l1 = rf_common.get_lambs(2.4e9, 10e6, nfft)
		psis = np.arange(-1,1, 0.1)
		distances = np.arange(0,200,1)
		ncols = len(psis)*len(distances)
		nrows = nfft*K
		X = np.zeros([nrows, ncols]).astype(np.complex)
		colID = 0
		for d in distances:
			for psi in psis:
				params = [[d],[1.0],[0],[psi]]
				ch= rf_common.get_chans_from_params(params, K, sep, l1)
				X[:, colID] = ch.ravel()
				colID += 1
	
		params = [[100],[1],[0],[0]]
		ch = rf_common.get_chans_from_params(params, K, sep, l1)
		ch = ch.ravel().reshape(1,-1)
		print(X.shape, ch.shape)
		temp = np.dot(ch.conj(), X)
		temp = np.abs(temp)
		print(temp.shape)
		temp = temp.ravel()
		temp = temp/np.max(temp)
		pl.plot(temp)
	pl.show()