import os
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')

import numpy as np
import pylab as pl
import scipy.io as sio

f = open("localization.txt",'r')
# f = open("new_2d_optml_res.txt",'r')
#f = open("combined_res.txt",'r')
#f = open("bounded_results.txt", 'r')
nnde = []
r2f2 = []
for line in f:
    line = line.strip()
    if line[0]=="#":
        continue
    line = line.replace("(","")
    line = line.replace(")","")
    line = line.replace(",","")
    line = line.replace(":","")
    line = line.split(" ")
    if line[0].find("beam1")>-1:
        if line[2]!="4":
            continue
        #if int(line[3])!=2:
        #    continue
        if line[0].find("nnde")>-1:
            nnde.append(line[1:])
        else:
            r2f2.append(line[1:])
    
f.close()

nnde = np.array(nnde).astype(float)
r2f2 = np.array(r2f2).astype(float)
bins = np.arange(20,26,0.15)


print(np.mean(nnde, 0))
print(np.mean(r2f2, 0))

# pl.hist(r2f2[:,8]*100)
# pl.show()
# exit()
bins = np.arange(0,100,1)
h6,b6,p1 = pl.hist(nnde[:,8]*100, bins, histtype="step", cumulative=True, normed=True)
h5,b5,p1 = pl.hist(r2f2[:,8]*100, bins, histtype="step", cumulative=True, normed=True)
b5 = b5[:-1]
b6 = b6[:-1]
print(np.percentile(nnde[:,8]*100, 50), np.percentile(r2f2[:,8]*100, 50))
pl.close()
pl.figure()
pl.plot(b5, h5, "b")
pl.plot(b6, h6, "r")
pl.xlabel("localization error (centimeter)", fontsize="x-large")
pl.ylabel("CDF", fontsize="x-large")
pl.xticks(fontsize="large")
pl.yticks(fontsize="large")
pl.tight_layout()
# pl.savefig("snr_nne.pdf")
pl.show()
exit()


h4,b4,p1 = pl.hist(r2f2[:,7], bins, histtype="step", cumulative=True, normed=True)
h3,b3,p1 = pl.hist(r2f2[:,6], bins, histtype="step", cumulative=True, normed=True)
h2,b2,p1 = pl.hist(r2f2[:,5], bins, histtype="step", cumulative=True, normed=True)
h1,b1,p1 = pl.hist(nnde[:,5], bins, histtype="step", cumulative=True, normed=True)

b4 = b4[:-1]
b3 = b3[:-1]
b2 = b2[:-1]
b1 = b1[:-1]
pl.close()
pl.figure(figsize=(5,3))
legs = []
#pl.plot(b4, h4, "^--", ms=7)
#legs.append("Best possible")

pl.plot(b2, h2, "v--", ms=7)
legs.append("R2F2")

pl.plot(b1, h1, "o--", ms=7)
legs.append("NNE+R2F2")

pl.plot(b3, h3, "*--", ms=7)
legs.append("No beam")

pl.xlabel("SNR, dB", fontsize="x-large")
pl.ylabel("CDF", fontsize="x-large")
pl.xticks(fontsize="large")
pl.yticks(fontsize="large")
pl.legend(legs, fontsize="large")
pl.tight_layout()
pl.savefig("snr_nne.pdf")
pl.show()
#exit()


#bins = np.arange(0, 25, 1)
#up = max(bins)
#nnde = np.where(nnde[:,0]>up, up, nnde[:,0])
#r2f2 = np.where(r2f2[:,0]>up, up, r2f2[:,0])
print(len(r2f2))
bins = 50
nnde = nnde[:,0]
r2f2 = r2f2[:,0]
h1,b1,p1 = pl.hist(nnde, bins, histtype="step", cumulative=True, normed=True)
h2,b2,p1 = pl.hist(r2f2, bins, histtype="step", cumulative=True, normed=True)
pl.close()

b2 = b2[:-1]
b1 = b1[:-1]

pl.figure(figsize=(5,3))
pl.plot(b2, h2, "v--", ms=7)
pl.plot(b1, h1, "o--", ms=7)
pl.ylim(0,1.05)
pl.xlabel("Runtime to prediction, seconds", fontsize="x-large")
pl.ylabel("CDF", fontsize="x-large")
pl.xticks(fontsize="large")
pl.yticks(fontsize="large")
pl.xscale('log')
pl.legend(["R2F2","NNE+R2F2"], fontsize="large", loc=2)
pl.tight_layout()
pl.savefig("runtime_nne.pdf")
pl.show()



