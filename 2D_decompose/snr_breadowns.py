import pylab as pl

optml_3 = [22.11, 22.05, 22.06, 21.65]
r2f2_3 = [21.53, 21.53, 21.46, 20.89]
opt_3 = [22.12, 22.18, 22.46, 22.24]
no_3 = [17.63,17.95,18.07,18.11]

pl.figure(figsize=(7,5))
leg = ["NNE+R2F2", "R2F2", "Oracle", "Baseline"]
pl.figure(figsize=(6,4))
pl.plot(optml_3, linewidth=3, marker="*", markersize=25)
pl.plot(r2f2_3, linewidth=3, marker="*", markersize=25)
pl.plot(opt_3, linewidth=3, marker="*", markersize=25)
pl.plot(no_3, linewidth=3, marker="*", markersize=25)
pl.legend(leg, fontsize="x-large")
pl.xticks([0,1,2,3],[2,4,5,6], fontsize="xx-large")
pl.yticks(fontsize="xx-large")
pl.xlabel("Number of multipath components", fontsize="xx-large")
pl.ylabel("Mean SNR, dB", fontsize="xx-large")
pl.tight_layout()
#pl.show()
pl.savefig("ant3_snr_v2.pdf")
pl.close()

