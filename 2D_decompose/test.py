import pickle
import collections
import numpy as np
import matplotlib.pyplot as pl

def deg2rad(x):
    return x/180*np.pi
def get_arrayfactor(theta, AWV):
    return np.dot(AWV, np.exp(1j*np.pi*np.arange(len(AWV))*np.cos(deg2rad(theta))))
def db(x):
    return 20*np.log10(x)
def db2mag(x):
    return 10**(x/20)

AWV=np.exp(-1j*np.pi*np.arange(8)*np.cos(deg2rad(45)))
thetas = np.arange(180)
ArrayFactor = np.array([get_arrayfactor(theta, AWV) for theta in np.arange(180)])
pl.figure()
pl.polar(deg2rad(np.arange(180)), np.where(db(ArrayFactor)>0,db(ArrayFactor),0) )

tmp = np.argwhere( np.abs(ArrayFactor) >= db2mag(-3)*np.max(np.abs(ArrayFactor)))
hpbw = max(tmp)-min(tmp)
pl.figure(6)
pl.plot(db(ArrayFactor)) 
# pl.hold(True)
pl.plot([min(tmp), max(tmp)], db([ArrayFactor[min(tmp)], ArrayFactor[max(tmp)]]))
pl.show()
exit()
# figure(7);
# polarplot(deg2rad(thetas),max(db(ArrayFactor),0));

with open('training_data/d0_1.5_30_aoastep10deg_num5e5/params_list.pckl', 'rb') as f:
    params_list = pickle.load(f)

cnt = collections.defaultdict(int)
ds =[]
psis =[]
for p in params_list:
    if len(p[0]) ==2:
        # ds.append( abs(p[0][1]-p[0][0]))
        # psis.append(abs(np.arccos(p[3][1])/np.pi*180-np.arccos(p[3][0])/np.pi*180))
        if 1< abs(p[0][1]-p[0][0])<=3 and 85< abs(np.arccos(p[3][1])/np.pi*180-np.arccos(p[3][0])/np.pi*180)<95:
            cnt[0]+=1
        # cnt[len(p[0])] +=1

# pl.figure()
# pl.hist(ds)
# pl.figure()
# pl.hist(psis)
# pl.show()
print(cnt)