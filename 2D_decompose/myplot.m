%%
load('/Users/wchen/Documents/github/optml2D/2D_decompose/sim_result.mat');
%%
figure(1);
cdfplot(beamform_snr_ground); hold on;
cdfplot(beamform_snr_optml2d); hold on;
cdfplot(beamform_snr_r2f2); hold on;
cdfplot(beamform_snr_mdtrack); hold on;
legend(["ground","optml2d","r2f2","mdtrack"]);
ylabel('SNR (dB)');

figure(2);
cdfplot(runtime_optml2d); hold on;
cdfplot(runtime_r2f2); hold on;
cdfplot(runtime_mdtrack); hold on;
legend(["optml2d","r2f2","mdtrack"]);
ylabel('Runtime (second)');

figure(3);
cdfplot(locerr_optml2d); hold on;
cdfplot(locerr_r2f2); hold on;
cdfplot(locerr_mdtrack); hold on;
legend(["optml2d","r2f2","mdtrack"]);
ylabel('Localization error (meter)');