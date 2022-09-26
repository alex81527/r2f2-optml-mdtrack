import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as pl
import  scipy.linalg as sl 
import os

class SAPC(object):
    @staticmethod
    def calc_tx_power(G, noise_vec, p, pmax):
        F = np.zeros(G.shape)
        for i in range(G.shape[0]):
            F[i,:] = G[i,:]/G[i,i]
            F[i,i] = 0.0
        
        v = noise_vec/np.diag(G)
        SIR = p/(np.matmul(F,p)+v)
        # print(SIR)
        iterations = 50
        p_list = [p]
        for _ in range(iterations):
            new_p = np.zeros(len(p))
            for i in range(len(p)):
                a = w[i]/(np.dot(w,F[:,i]*SIR/p_list[-1]))
                b = w[i]*F[i,i]*SIR[i]/p_list[-1][i]
                new_p[i] = min(a - b, pmax)
            
            SIR = new_p/(np.matmul(F,new_p)+v)
            p_list.append(new_p)
        
        return p_list

    def perron(A, which='right'):
        row, col = A.shape
        if np.any(A<0):
            raise Exception('Input matrix should be non-negative.')
        if np.max(A+np.eye(row)**(row-1))<=0:
            raise Exception('Input matrix should be irreducible.')
        if which=='left':
            A = A.T
        
        iterated = np.random.uniform(size=(row,1))
        # normalization
        A = np.linalg.inv(np.diag(np.diag(A)))*A
        # print('A', A)
        lamb = np.max(np.sum(A,axis=0))
        err = 1
        eps = 2.2204e-16
        if row>1:
            while err>eps:
                B = np.linalg.inv(lamb*np.eye(row)-A)
                p, l, u = sl.lu(lamb*np.eye(row)-A)
                L = np.matrix(np.matmul(p,l))
                U = np.matrix(u)
                iterated = np.linalg.solve(L*U,iterated)
                p = iterated/(B*iterated)
                lamb_max = lamb - np.min(p)
                lamb_min = lamb - np.max(p)
                err = (lamb_max-lamb_min)/lamb_max
                iterated = np.linalg.solve(L*U,iterated)
                lamb = lamb_max
                # print(err)
        
        perron_root = lamb
        perron_vector = iterated/np.sum(iterated)
        return perron_root, perron_vector


def worker(x,y,z):
    return x+2+y+z

if __name__ == "__main__":
    # B = np.matrix([[0.001148484553124,    0.243073397420598],
    #             [0.025234858750195,    0.003851515446876]])
    # B = np.matrix([[0.03902041, 0.28094533],[0.28094533, 0.03902041]])
    # print(B)
    # r, rv = SAPC.perron(B, 'right')
    # l, lv = SAPC.perron(B, 'left')
    # print(r, rv)
    # print(l, lv)
    # print(np.squeeze(rv*lv))   
    # ps = Pool(4)
    # res = [ps.apply_async(worker, args=(x,1,2)) for x in range(100)]
    # print([r.get() for r in res])
    # n = os.fork()
    # if n>0:
    #     print('parent', os.getpid())
    # else:
    #     print('child', os.getpid())
    # print(os.cpu_count())
    # orig_dbm = np.array([])
    np.isnan
    exit()
    
    G = np.array([[0.0315,    0.0076,   0.0144],
    [0.0315,   0.3559,    0.0144],
    [0.0076,    0.0063,    0.0861]])
    G*=1e-7
    F = np.zeros(G.shape)
    for i in range(G.shape[0]):
        F[i,:] = G[i,:]/G[i,i]
        F[i,i] = 0.0
    w = np.array([1, 0.5, 0.5])
    n = np.ones(3)*1e-9
    v = n/np.diag(G)
    p = np.ones(3)*100
    res = SAPC.calc_tx_power(G,n,p,100)

    exh = np.zeros((100,100,100))
    i = 0
    for xx in range(0,100,5):
        ii=0
        for yy in range(0,100,5):
            iii=0
            for zz in range(100):
                p = [xx,yy,zz]
                SIR = p/(np.matmul(F,p)+v)
                exh[i,ii,iii] = np.dot(w, np.log10(1+SIR) )
                iii+=1
            ii+=1
        i+=1

    ind = np.unravel_index(np.argmax(exh, axis=None), exh.shape)
    print(ind)
    pl.figure()
    pl.plot(res)
    pl.show()