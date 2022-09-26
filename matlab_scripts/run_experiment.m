%% initialization, run it once
clear;
nodes = wl_initNodes(2)
ifc_ids = wl_getInterfaceIDs(nodes(1));
sts_f = zeros(1,64);
sts_f(1:27) = [0 0 0 0 -1-1i 0 0 0 -1-1i 0 0 0 1+1i 0 0 0 1+1i 0 0 0 1+1i 0 0 0 1+1i 0 0];
sts_f(39:64) = [0 0 1+1i 0 0 0 -1-1i 0 0 0 1+1i 0 0 0 -1-1i 0 0 0 -1-1i 0 0 0 1+1i 0 0 0];
sts_t = ifft(sqrt(13/6).*sts_f, 64);
sts_t = sts_t(1:16);
lts_f = [0 1 -1 -1 1 1 -1 1 -1 1 -1 -1 -1 -1 -1 1 1 -1 -1 1 -1 1 -1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 1 1 -1 -1 1 1 -1 1 -1 1 1 1 1 1 1 -1 -1 1 1 -1 1 -1 1 1 1 1];
lts_t = ifft(lts_f, 64);
preamble = [repmat(sts_t, 1, 30)  lts_t(33:64) lts_t lts_t];
% preamble = [lts_t(33:64) lts_t lts_t];
preamble_interval = 25e-3; % lts every 25 ms
transmission_time = 1;     % 1 second
preamble_train = [preamble zeros(1, preamble_interval*20e6 - length(preamble))];
preamble_train = repmat(preamble_train, 1, ceil(transmission_time/preamble_interval));
%% node sync
USE_EXTERNAL_TRIGGER = true;
USE_AGC = true;
ManualTxGainRF = 10;                                % [0,63] for [0:31] dB RF gain
ManualTxGainBB = 1;                                  % [0,1,2,3] for [-5, -3,-1.5, 0] dB
ManualRxGainRF = [3 3 3 3];                   % Rx RF Gain in [1:3] (ignored if USE_AGC is true)
ManualRxGainBB = [16, 16, 16, 16]-1;                   % Rx Baseband Gain in [0:31] (ignored if USE_AGC is true)
RF_TX = ifc_ids.RF_A;                    % Transmit RF interface
RF_TX_VEC = ifc_ids.RF_A;
RF_RX = ifc_ids.RF_ALL;                    % Receive RF interface
RF_RX_VEC = ifc_ids.RF_ALL_VEC;                 % Vector version of transmit RF interface
BAND           = 2.4;
CHANNEL        = 1;
% siso ofdm
USE_WARPLAB_TXRX        = 1;           % Enable WARPLab-in-the-loop (otherwise sim-only)
TX_SCALE                = 1.0;          % Scale for Tx waveform ([0:1])
SC_IND_PILOTS           = [8 22 44 58];                           % Pilot subcarrier indices
SC_IND_DATA             = [2:7 9:21 23:27 39:43 45:57 59:64];     % Data subcarrier indices
N_SC                    = 64;                                     % Number of subcarriers
CP_LEN                  = 16;                                     % Cyclic prefix length
INTERP_RATE             = 2;                                      % Interpolation rate (must be 2)
FFT_OFFSET                    = 0;           % Number of CP samples to use in FFT (on average)
LTS_CORR_THRESH               = 0.8;         % Normalized threshold for LTS correlation
DO_APPLY_CFO_CORRECTION       = 1;           % Enable CFO estimation/correction
DO_APPLY_PHASE_ERR_CORRECTION = 1;           % Enable Residual CFO estimation/correction
DO_APPLY_SFO_CORRECTION       = 1;           % Enable SFO estimation/correction
DECIMATE_RATE                 = INTERP_RATE;
MAX_TX_LEN              = 56e6;        % Maximum number of samples to use for this experiment
nodes(1).baseband.MEX_TRANSPORT_MAX_IQ = 50e6;
nodes(2).baseband.MEX_TRANSPORT_MAX_IQ = 50e6;


for iter = 1:1
    phase_diff = zeros(2, length(RF_RX_VEC));
    for i = 1: size(phase_diff, 1)
        phase_diff(i, :) = wl_example_siso_txrx_nodeSync(nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ...
            ManualRxGainRF, ManualRxGainBB, RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL);
    end

    phase_diff;
%     pause(1)
end

if USE_AGC
        s=wl_basebandCmd(nodes(2), [RF_RX_VEC], 'agc_state');
        fprintf('agc_done_addr: %d\n', wl_basebandCmd(nodes(2), 'agc_done_addr') );
        for k=1:length(RF_RX_VEC)
            fprintf('RF %d [AGC] RF: %d, BB: %d, RSSI: %d\n',k, s(1,k),s(2,k),s(3,k));
        end
end
%% RFA 
ManualRxGainRF = 2;                   % Rx RF Gain in [1:3] (ignored if USE_AGC is true)
ManualRxGainBB = 5;                  % Rx Baseband Gain in [0:31] (ignored if USE_AGC is true)
RF_TX          = ifc_ids.RF_A;                    % Transmit RF interface
RF_TX_VEC      = ifc_ids.RF_A;
RF_RX          = ifc_ids.RF_A;                    % Receive RF interface
RF_RX_VEC      = ifc_ids.RF_A; 
clear cor hist_tof; HA = zeros(64,1);
for ii =1:30
    H = wl_example_siso_ofdm_txrx(preamble, nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ...
            ManualRxGainRF, ManualRxGainBB, RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL, ...
            USE_WARPLAB_TXRX, TX_SCALE, SC_IND_PILOTS, SC_IND_DATA, N_SC, CP_LEN, INTERP_RATE, FFT_OFFSET, ...
            LTS_CORR_THRESH, DO_APPLY_CFO_CORRECTION, DO_APPLY_PHASE_ERR_CORRECTION, DO_APPLY_SFO_CORRECTION, ...
            MAX_TX_LEN);
    HA = HA + H;
    figure(3);
    subplot(2,1,1); plot(unwrap(angle(fftshift(H)))); hold on;
    subplot(2,1,2);
    tofs = -4*500:500*2;
    for jj=1:length(tofs)
        cor(jj) = abs(sum(H.*exp(1j.*2.*pi.*(tofs(jj)/500)./64.*[0:63].')))^2;
    end
    plot(tofs.*0.1, cor); hold on; xlabel('ToF (ns)');
    [M, I] = max(cor); fprintf('Measured tof %.2f ns\n', tofs(I)*0.1);
    hist_tof(ii) = tofs(I)*0.1;
end
HA = HA./ii;
figure(21); hist(hist_tof);
%% RFB 
ManualRxGainRF = 2;                   % Rx RF Gain in [1:3] (ignored if USE_AGC is true)
ManualRxGainBB = 5;                  % Rx Baseband Gain in [0:31] (ignored if USE_AGC is true)
RF_TX          = ifc_ids.RF_A;                    % Transmit RF interface
RF_TX_VEC      = ifc_ids.RF_A;
RF_RX          = ifc_ids.RF_B;                    % Receive RF interface
RF_RX_VEC      = ifc_ids.RF_B; 
clear cor; HB = zeros(64,1);
for ii =1:10
    H = wl_example_siso_ofdm_txrx(preamble, nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ...
            ManualRxGainRF, ManualRxGainBB, RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL, ...
            USE_WARPLAB_TXRX, TX_SCALE, SC_IND_PILOTS, SC_IND_DATA, N_SC, CP_LEN, INTERP_RATE, FFT_OFFSET, ...
            LTS_CORR_THRESH, DO_APPLY_CFO_CORRECTION, DO_APPLY_PHASE_ERR_CORRECTION, DO_APPLY_SFO_CORRECTION, ...
            MAX_TX_LEN);
    HB = HB + H;
    figure(3);
    subplot(2,1,1); plot(unwrap(angle(fftshift(H))), 'r'); hold on;
    subplot(2,1,2);
    tofs = -4*500:500*2;
    for jj=1:length(tofs)
        cor(jj) = abs(sum(H.*exp(1j.*2.*pi.*(tofs(jj)/500)./64.*[0:63].')))^2;
    end
    plot(tofs.*0.1, cor, 'r'); hold on; xlabel('ToF (ns)');
    [M, I] = max(cor); fprintf('Measured tof %.2f ns\n', tofs(I)*0.1);
end
HB = HB./ii;
%% RFC
ManualRxGainRF = 2;                   % Rx RF Gain in [1:3] (ignored if USE_AGC is true)
ManualRxGainBB = 5;                  % Rx Baseband Gain in [0:31] (ignored if USE_AGC is true)
RF_TX          = ifc_ids.RF_A;                    % Transmit RF interface
RF_TX_VEC      = ifc_ids.RF_A;
RF_RX          = ifc_ids.RF_C;                    % Receive RF interface
RF_RX_VEC      = ifc_ids.RF_C; 
clear cor; HC = zeros(64,1);
for ii =1:10
    H = wl_example_siso_ofdm_txrx(preamble, nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ...
            ManualRxGainRF, ManualRxGainBB, RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL, ...
            USE_WARPLAB_TXRX, TX_SCALE, SC_IND_PILOTS, SC_IND_DATA, N_SC, CP_LEN, INTERP_RATE, FFT_OFFSET, ...
            LTS_CORR_THRESH, DO_APPLY_CFO_CORRECTION, DO_APPLY_PHASE_ERR_CORRECTION, DO_APPLY_SFO_CORRECTION, ...
            MAX_TX_LEN);
    HC = HC + H;
    figure(3);
    subplot(2,1,1); plot(unwrap(angle(fftshift(H))), 'g'); hold on;
    subplot(2,1,2);
    tofs = -4*500:500*2;
    for jj=1:length(tofs)
        cor(jj) = abs(sum(H.*exp(1j.*2.*pi.*(tofs(jj)/500)./64.*[0:63].')))^2;
    end
    plot(tofs.*0.1, cor, 'g'); hold on; xlabel('ToF (ns)');
    [M, I] = max(cor); fprintf('Measured tof %.2f ns\n', tofs(I)*0.1);
end
HC = HC./ii;
%% RFD 
ManualRxGainRF = 2;                   % Rx RF Gain in [1:3] (ignored if USE_AGC is true)
ManualRxGainBB = 5;                  % Rx Baseband Gain in [0:31] (ignored if USE_AGC is true)
RF_TX          = ifc_ids.RF_A;                    % Transmit RF interface
RF_TX_VEC      = ifc_ids.RF_A;
RF_RX          = ifc_ids.RF_D;                    % Receive RF interface
RF_RX_VEC      = ifc_ids.RF_D; 
clear cor; HD = zeros(64,1);
for ii =1:10
    H = wl_example_siso_ofdm_txrx(preamble, nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ...
            ManualRxGainRF, ManualRxGainBB, RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL, ...
            USE_WARPLAB_TXRX, TX_SCALE, SC_IND_PILOTS, SC_IND_DATA, N_SC, CP_LEN, INTERP_RATE, FFT_OFFSET, ...
            LTS_CORR_THRESH, DO_APPLY_CFO_CORRECTION, DO_APPLY_PHASE_ERR_CORRECTION, DO_APPLY_SFO_CORRECTION, ...
            MAX_TX_LEN);
    HD = HD + H;
    figure(3);
    subplot(2,1,1); plot(unwrap(angle(fftshift(H))), 'k'); hold on;
    subplot(2,1,2);
    tofs = -4*500:500*2;
    for jj=1:length(tofs)
        cor(jj) = abs(sum(H.*exp(1j.*2.*pi.*(tofs(jj)/500)./64.*[0:63].')))^2;
    end
    plot(tofs.*0.1, cor, 'k'); hold on; xlabel('ToF (ns)');
    [M, I] = max(cor); fprintf('Measured tof %.2f ns\n', tofs(I)*0.1);
end
HD = HD./ii;
%% between RF
ManualTxGainRF = 10;                                % [0,63] for [0:31] dB RF gain
ManualTxGainBB = 1;                                  % [0,1,2,3] for [-5, -3,-1.5, 0] dB
ManualRxGainRF = [3 3 3 3];                   % Rx RF Gain in [1:3] (ignored if USE_AGC is true)
ManualRxGainBB = [16, 16, 16, 16]-8;                   % Rx Baseband Gain in [0:31] (ignored if USE_AGC is true)
USE_AGC = 0;
DO_APPLY_CFO_CORRECTION = false;
RF_TX          = ifc_ids.RF_A;                    % Transmit RF interface
RF_TX_VEC = ifc_ids.RF_A;
RF_RX         = ifc_ids.RF_ALL;                    % Receive RF interface
RF_RX_VEC  = ifc_ids.RF_ALL_VEC;
avg_H = zeros(64,4);
for ll=1:10
    H = wl_example_siso_ofdm_txrx(preamble, nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ...
                ManualRxGainRF, ManualRxGainBB, ManualTxGainRF, ManualTxGainBB, RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL, ...
                USE_WARPLAB_TXRX, TX_SCALE, SC_IND_PILOTS, SC_IND_DATA, N_SC, CP_LEN, INTERP_RATE, FFT_OFFSET, ...
                LTS_CORR_THRESH, DO_APPLY_CFO_CORRECTION, DO_APPLY_PHASE_ERR_CORRECTION, DO_APPLY_SFO_CORRECTION, ...
                MAX_TX_LEN);
    avg_H = avg_H + H;
end
H = avg_H./ll;

 if USE_AGC
    s=wl_basebandCmd(nodes(2), [RF_RX_VEC], 'agc_state');
    fprintf('agc_done_addr: %d\n', wl_basebandCmd(nodes(2), 'agc_done_addr') );
    for k=1:length(RF_RX_VEC)
        fprintf('RF %d [AGC] RF: %d, BB: %d, RSSI: %d\n',k, s(1,k),s(2,k),s(3,k));
        rf_gain = [0, 15,30];
        bb_gain_step= 63/31;
        gain_db(k) = rf_gain(s(1,k))+bb_gain_step*s(2, k);
    end
 else
     for k=1:length(RF_RX_VEC)
%         fprintf('RF %d [AGC] RF: %d, BB: %d, RSSI: %d\n',k, s(1,k),s(2,k),s(3,k));
        rf_gain = [0, 15,30];
        bb_gain_step= 63/31;
        gain_db(k) = rf_gain(ManualRxGainRF(k))+bb_gain_step*ManualRxGainBB(k);
    end
 end

% H = H.* INTER_RF_CALIBRATION;
figure(10);
color = ['b', 'r', 'g', 'k'];
leg = ['A', 'B', 'C', 'D'];
for ii=1:length(RF_RX_VEC)
    subplot(2,2,1);
    plot(db(nonzeros(fftshift(H(:, ii)))), color(ii)); hold on; ylabel('dB'); title('AGC enabled');
    subplot(2,2,3);
    plot(angle(nonzeros(fftshift(H(:, ii)))), color(ii)); hold on; ylabel('radian');
    subplot(2,2,2);
    plot(db(nonzeros(fftshift(H(:, ii)))) - gain_db(ii), color(ii)); hold on; ylabel('dB'); title('AGC gain removed');
    subplot(2,2,4);
    plot(angle(nonzeros(fftshift(H(:, ii)))), color(ii)); hold on; ylabel('radian');
end
legend('A', 'B', 'C', 'D');

INTER_RF_CALIBRATION = ones(64, 4);
clear attenuation_db attenuation_diff_db phase_diff
for ii=1:length(RF_RX_VEC)
    for jj=1:64
        if H(jj,ii)==0
            continue
        end
        attenuation_db(jj,ii) = db(H(jj, ii)) - gain_db(ii);
        attenuation_diff_db(jj, ii) = attenuation_db(jj,ii) - attenuation_db(jj,1);

        phase_diff(jj, ii) = angle(H(jj, ii)/H(jj,1));
        
%         INTER_RF_CALIBRATION(jj,ii) = db2mag(-attenuation_diff_db(jj,ii))*exp(-1j*phase_diff(jj, ii));
    end
end

figure(2);
subplot(2,1,1); plot(1:64, -attenuation_diff_db); hold on; ylabel('dB'); title('gain correction value, calibrated to RFA');
subplot(2,1,2); plot(1:64, -phase_diff); hold on; ylabel('radian'); title('phase correction value, calibrated to RFA');
legend('A', 'B', 'C', 'D');

avg_attenuation_diff_db = sum(attenuation_diff_db)./52;
avg_phase_diff = sum(phase_diff)./52;
fprintf('RF chain avg gain difference (dB): %f %f %f %f \n', avg_attenuation_diff_db);
fprintf('RF chain avg phase difference (radian): %f %f %f %f \n', avg_phase_diff);
INTER_RF_CALIBRATION = repmat(db2mag(-avg_attenuation_diff_db).*exp(-1j*avg_phase_diff), 64 ,1);

H = wl_example_siso_ofdm_txrx(preamble, nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ...
                ManualRxGainRF, ManualRxGainBB, ManualTxGainRF, ManualTxGainBB, RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL, ...
                USE_WARPLAB_TXRX, TX_SCALE, SC_IND_PILOTS, SC_IND_DATA, N_SC, CP_LEN, INTERP_RATE, FFT_OFFSET, ...
                LTS_CORR_THRESH, DO_APPLY_CFO_CORRECTION, DO_APPLY_PHASE_ERR_CORRECTION, DO_APPLY_SFO_CORRECTION, ...
                MAX_TX_LEN);
H = H.* INTER_RF_CALIBRATION;
figure(11);
color = ['b', 'r', 'g', 'k'];
for ii=1:length(RF_RX_VEC)
    subplot(2,2,1);
    plot(db(nonzeros(fftshift(H(:, ii)))), color(ii)); hold on; ylabel('dB'); title('AGC enabled');
    subplot(2,2,3);
    plot(angle(nonzeros(fftshift(H(:, ii)))), color(ii)); hold on; ylabel('radian');
    subplot(2,2,2);
    plot(db(nonzeros(fftshift(H(:, ii)))) - gain_db(ii), color(ii)); hold on; ylabel('dB'); title('AGC gain removed');
    subplot(2,2,4);
    plot(angle(nonzeros(fftshift(H(:, ii)))), color(ii)); hold on; ylabel('radian');
end
legend('A', 'B', 'C', 'D');         
% INTER_RF_CALIBRATION = ones(64, 4);
% for ii=1:64
%     if HA(ii)==0
%         diff_AB(ii) = 1;
%         diff_AC(ii) = 1;
%         diff_AD(ii) = 1;
%         INTER_RF_CALIBRATION(ii, :) = ones(1, 4);
%     else
%         diff_AB(ii) = HA(ii)./HB(ii);
%         diff_AC(ii) = HA(ii)./HC(ii);
%         diff_AD(ii) = HA(ii)./HD(ii);
% %         INTER_RF_CALIBRATION(ii, :) = [1/HA(ii) 1/HB(ii) 1/HC(ii) 1/HD(ii)];
%         INTER_RF_CALIBRATION(ii, :) = [1 exp(1j*angle(diff_AB(ii))) exp(1j*angle(diff_AC(ii))) exp(1j*angle(diff_AD(ii)))];
%     end
% end
% mean(nonzeros(angle(diff_AB)))
% mean(nonzeros(angle(diff_AC)))
% mean(nonzeros(angle(diff_AD)))
% % tmp = repmat([1 exp(1j*1.6651) exp(1j*-0.0353) exp(1j*-0.7435)], 64, 1);
% figure(6);
% % plot(1:64, angle([HA HB.*diff_AB.' HC HD]));
% % plot(1:64, angle([HA HB HC HD].*INTER_RF_CALIBRATION));
% plot(1:64, angle([diff_AB.' diff_AC.' diff_AD.']));
% axis([1 64 -pi pi]);
%% measure noise
% ManualRxGainRF = [3 3 3 3]-2;                   % Rx RF Gain in [1:3] (ignored if USE_AGC is true)
% ManualRxGainBB = [16, 16, 16, 16]-0;                   % Rx Baseband Gain in [0:31] (ignored if USE_AGC is true)
USE_AGC = 0;
RF_TX          = ifc_ids.RF_A;                    % Transmit RF interface
RF_TX_VEC = ifc_ids.RF_A;
RF_RX         = ifc_ids.RF_ALL;                    % Receive RF interface
RF_RX_VEC  = ifc_ids.RF_ALL_VEC;

figure(9);
for ii=1:3
    for jj=0:31
        ManualRxGainRF = ones(1,4)*ii;                   % Rx RF Gain in [1:3] (ignored if USE_AGC is true)
        ManualRxGainBB = ones(1,4)*jj;
        noise = wl_example_siso_ofdm_txrx_get_noise(zeros(1, 4000), nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ...
                    ManualRxGainRF, ManualRxGainBB, RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL, ...
                    USE_WARPLAB_TXRX, TX_SCALE, SC_IND_PILOTS, SC_IND_DATA, N_SC, CP_LEN, INTERP_RATE, FFT_OFFSET, ...
                    LTS_CORR_THRESH, DO_APPLY_CFO_CORRECTION, DO_APPLY_PHASE_ERR_CORRECTION, DO_APPLY_SFO_CORRECTION, ...
                    MAX_TX_LEN);
        noise_pwr(jj+1, :) = noise;
        noisepwr_lut(ii,jj+1,:) = noise;
    end
    subplot(3,1,ii);
    plot(0:31, noise_pwr);
%     plot(0:31, noise_pwr -  squeeze(noisepwr_lut(ii,:,:)));
end
legend('A', 'B','C','D');
%% collect data
clear noise_pwr;
ManualTxGainRF = 10;                                % [0,63] for [0:31] dB RF gain
ManualTxGainBB = 1;                                  % [0,1,2,3] for [-5, -3,-1.5, 0] dB
ManualRxGainRF = [3 3 3 3];                   % Rx RF Gain in [1:3] (ignored if USE_AGC is true)
ManualRxGainBB = [16, 16, 16, 16]-8;                   % Rx Baseband Gain in [0:31] (ignored if USE_AGC is true)
USE_AGC = 1;
CHANNEL        = 5;
RF_TX          = ifc_ids.RF_A;                    % Transmit RF interface
RF_TX_VEC = ifc_ids.RF_A;
RF_RX         = ifc_ids.RF_ALL;                    % Receive RF interface
RF_RX_VEC  = ifc_ids.RF_ALL_VEC;
DO_APPLY_CFO_CORRECTION = false;

% wl_example_siso_txrx_nodeSync(nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ...
% ManualRxGainRF, ManualRxGainBB, RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL);

% noise = wl_example_siso_ofdm_txrx_get_noise(zeros(1, 4000), nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ...
%             ManualRxGainRF, ManualRxGainBB, RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL, ...
%             USE_WARPLAB_TXRX, TX_SCALE, SC_IND_PILOTS, SC_IND_DATA, N_SC, CP_LEN, INTERP_RATE, FFT_OFFSET, ...
%             LTS_CORR_THRESH, DO_APPLY_CFO_CORRECTION, DO_APPLY_PHASE_ERR_CORRECTION, DO_APPLY_SFO_CORRECTION, ...
%             MAX_TX_LEN)


clear cor; 
for ii =1:1
    H = wl_example_siso_ofdm_txrx(preamble, nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ...
                ManualRxGainRF, ManualRxGainBB, ManualTxGainRF, ManualTxGainBB, RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL, ...
                USE_WARPLAB_TXRX, TX_SCALE, SC_IND_PILOTS, SC_IND_DATA, N_SC, CP_LEN, INTERP_RATE, FFT_OFFSET, ...
                LTS_CORR_THRESH, DO_APPLY_CFO_CORRECTION, DO_APPLY_PHASE_ERR_CORRECTION, DO_APPLY_SFO_CORRECTION, ...
                MAX_TX_LEN);
        
    if USE_AGC
        s=wl_basebandCmd(nodes(2), [RF_RX_VEC], 'agc_state');
        fprintf('agc_done_addr: %d\n', wl_basebandCmd(nodes(2), 'agc_done_addr') );
        for k=1:length(RF_RX_VEC)
            fprintf('RF %d [AGC] RF: %d, BB: %d, RSSI: %d\n',k, s(1,k),s(2,k),s(3,k));
            rf_gain = [0, 15,30];
            bb_gain_step= 63/31;
            gain_db(k) = rf_gain(s(1,k))+bb_gain_step*s(2, k);
            fprintf('RF %d noise power: %f\n', k, noisepwr_lut(s(1,k), s(2,k)+1, k));
            noise_pwr(k) = noisepwr_lut(s(1,k), s(2,k)+1, k);
        end
     else
         for k=1:length(RF_RX_VEC)
    %         fprintf('RF %d [AGC] RF: %d, BB: %d, RSSI: %d\n',k, s(1,k),s(2,k),s(3,k));
            rf_gain = [0, 15,30];
            bb_gain_step= 63/31;
            gain_db(k) = rf_gain(ManualRxGainRF(k))+bb_gain_step*ManualRxGainBB(k);
            fprintf('RF %d noise power: %f\n', k, noisepwr_lut(ManualRxGainRF(k), ManualRxGainBB(k)+1, k));
            noise_pwr(k) = noisepwr_lut(ManualRxGainRF(k), ManualRxGainBB(k)+1, k);
        end
    end
    
    calibrated_H = H .* INTER_RF_CALIBRATION(:, 1:size(H,2));
%     calibrated_H = H .* INTER_RF_CALIBRATION(:, 1:size(H,2)).*repmat(db2mag(-gain_db), 64, 1);
    adjusted_noisepwr = noise_pwr.*db2pow(db(INTER_RF_CALIBRATION(1,:)));
    calibrated_H = calibrated_H(:, [1,3,4]);
    adjusted_noisepwr = adjusted_noisepwr([1,3,4]);

    fprintf('estimated snr (dB): %f %f %f\n', 10*log10(mean(abs(ifft(calibrated_H)).^2)./adjusted_noisepwr));
    save(sprintf('./experiment_data/calibrated_H_%d.mat', ii), 'calibrated_H');
    save(sprintf('./experiment_data/adjusted_noisepwr_%d.mat', ii), 'adjusted_noisepwr');
%     show_heatmap(H);
    fprintf('Iteration %d, \n', ii);
    show_heatmap(calibrated_H);
%     pause(0.1);
    
%     cor_H = calibrated_H;
%     figure(200);
%     for jj=1:4
%         subplot(2,1,1); plot(unwrap(angle(fftshift(cor_H(:,jj))))); hold on;
%         subplot(2,1,2);
%         tofs = -3*500:500*3;
%         for kk=1:length(tofs)
%             cor(kk) = abs(sum(cor_H(:,jj).*exp(1j.*2.*pi.*(tofs(kk)/500)./64.*[0:63].')))^2;
%         end
%         plot(tofs.*0.1, cor); hold on; xlabel('ToF (ns)');
%         [M, I] = max(cor); fprintf('Measured tof %.2f ns\n', tofs(I)*0.1);
%     end

    plot_H = H .* INTER_RF_CALIBRATION(:, 1:size(H,2));
    figure(10);
    color = ['b', 'r', 'g', 'k'];
    leg = ['A', 'B', 'C', 'D'];
    for ii=1:size(H,2)
        subplot(2,2,1);
        plot(db(nonzeros(fftshift(plot_H(:, ii)))), color(ii)); hold on; ylabel('dB'); title('AGC enabled');
        subplot(2,2,3);
        plot(angle(nonzeros(fftshift(plot_H(:, ii)))), color(ii)); hold on; ylabel('radian');
        subplot(2,2,2);
        plot(db(nonzeros(fftshift(plot_H(:, ii)))) - gain_db(ii), color(ii)); hold on; ylabel('dB'); title('AGC gain removed');
        subplot(2,2,4);
        plot(angle(nonzeros(fftshift(plot_H(:, ii)))), color(ii)); hold on; ylabel('radian');
    end
%     legend('A', 'B', 'C', 'D');
    legend('A', 'C', 'D');
end
%% multi preambles
% preamble_interval = 25e-3; % lts every 25 ms
% transmission_time = 1;     % 1 second
% preamble_train = [preamble zeros(1, preamble_interval*20e6 - length(preamble))];
% preamble_train = repmat(preamble_train, 1, ceil(transmission_time/preamble_interval));
% preamble_train = [preamble 0.1+0.05.*rand(1, preamble_interval*20e6 - length(preamble)).*exp(1j.*2.*pi.*rand(1, preamble_interval*20e6 - length(preamble)))];
% preamble_train = repmat(preamble_train, 1, ceil(transmission_time/preamble_interval));
ManualTxGainRF = 10;                                % [0,63] for [0:31] dB RF gain
ManualTxGainBB = 1;                                  % [0,1,2,3] for [-5, -3,-1.5, 0] dB
ManualRxGainRF = [3 3 3 3];                   % Rx RF Gain in [1:3] (ignored if USE_AGC is true)
ManualRxGainBB = [16, 16, 16, 16]-8;                   % Rx Baseband Gain in [0:31] (ignored if USE_AGC is true)
USE_AGC = 1;
CHANNEL = 5;
DO_APPLY_CFO_CORRECTION = false;
for ii=1:1
    H = wl_example_siso_ofdm_txrx_preamble_train(preamble_train, nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ...
            ManualRxGainRF, ManualRxGainBB, ManualTxGainRF, ManualTxGainBB,RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL, ...
            USE_WARPLAB_TXRX, TX_SCALE, SC_IND_PILOTS, SC_IND_DATA, N_SC, CP_LEN, INTERP_RATE, FFT_OFFSET, ...
            LTS_CORR_THRESH, DO_APPLY_CFO_CORRECTION, DO_APPLY_PHASE_ERR_CORRECTION, DO_APPLY_SFO_CORRECTION, ...
            MAX_TX_LEN, preamble_interval, transmission_time);
    
    if USE_AGC
        s=wl_basebandCmd(nodes(2), [RF_RX_VEC], 'agc_state');
        fprintf('agc_done_addr: %d\n', wl_basebandCmd(nodes(2), 'agc_done_addr') );
        for k=1:length(RF_RX_VEC)
            fprintf('RF %d [AGC] RF: %d, BB: %d, RSSI: %d\n',k, s(1,k),s(2,k),s(3,k));
            rf_gain = [0, 15,30];
            bb_gain_step= 63/31;
            gain_db(k) = rf_gain(s(1,k))+bb_gain_step*s(2, k);
            fprintf('RF %d noise power: %f\n', k, noisepwr_lut(s(1,k), s(2,k)+1, k));
            noise_pwr(k) = noisepwr_lut(s(1,k), s(2,k)+1, k);
        end
     else
         for k=1:length(RF_RX_VEC)
    %         fprintf('RF %d [AGC] RF: %d, BB: %d, RSSI: %d\n',k, s(1,k),s(2,k),s(3,k));
            rf_gain = [0, 15,30];
            bb_gain_step= 63/31;
            gain_db(k) = rf_gain(ManualRxGainRF(k))+bb_gain_step*ManualRxGainBB(k);
            fprintf('RF %d noise power: %f\n', k, noisepwr_lut(ManualRxGainRF(k), ManualRxGainBB(k)+1, k));
            noise_pwr(k) = noisepwr_lut(ManualRxGainRF(k), ManualRxGainBB(k)+1, k);
        end
    end
        
        
    for jj=1:length(H)
        calibrated_H = H{jj} .* INTER_RF_CALIBRATION(:, 1:size(H{jj},2));
        adjusted_noisepwr = noise_pwr.*db2pow(db(INTER_RF_CALIBRATION(1,:)));
        calibrated_H = calibrated_H(:, [1,3,4]);
        adjusted_noisepwr = adjusted_noisepwr([1,3,4]);
        fprintf('noisepwr after calibration: %f %f %f %f\n', adjusted_noisepwr);
        fprintf('estimated snr (dB): %f %f %f %f\n', 10*log10(mean(abs(ifft(calibrated_H)).^2)./adjusted_noisepwr));
        save(sprintf('./experiment_data/calibrated_H_%d.mat', jj), 'calibrated_H');
        save(sprintf('./experiment_data/adjusted_noisepwr_%d.mat', jj), 'adjusted_noisepwr');
    end
end

length(H)

clear dop_H cor;
for ii=1:100
    load(sprintf('./experiment_data/preamble_train_human_walking/calibrated_H_%d.mat', ii));
    load(sprintf('./experiment_data/preamble_train_human_walking/adjusted_noisepwr_%d.mat', ii));
    snr = 10*log10(mean(abs(ifft(calibrated_H)).^2)./adjusted_noisepwr);
    fprintf('est snr %f %f %f\n', snr);
    %figure(10);plot(1:64, abs(calibrated_H)); hold on;%input('');
    figure(10);plot(1:3, snr); hold on;%input('');
    % show_heatmap(calibrated_H);
%     input('');
%     pause(0.1);
    dop_H(:,ii) = ifft(calibrated_H(:,1));
end

dop_H = dop_H(:).';
dops = [-50:50]+200;
for ii=1:length(dops)
    dop_vec = zeros(1, transmission_time/preamble_interval*64);
    for jj=1:transmission_time/preamble_interval
        dop_vec((jj-1)*64+[1:64]) = exp(1j.*2.*pi.*dops(ii)./20e6.* ((jj-1)*(preamble_interval*20e6)+[0:63]));
    end
    
    cor(ii) = dot(dop_vec, dop_H);
end
figure;
stem(dops-200, abs(cor));
xlabel('Doppler shift (Hz)');
title('preamble_train_antenna_moving_close');

%% 1. calibration between RF chains, need to connect them through wires
avg_H = zeros(64, 4);
figure(2);
for ii=1:10   
    H = wl_example_siso_ofdm_txrx(preamble, nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ...
            ManualRxGainRF, ManualRxGainBB, RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL, ...
            USE_WARPLAB_TXRX, TX_SCALE, SC_IND_PILOTS, SC_IND_DATA, N_SC, CP_LEN, INTERP_RATE, FFT_OFFSET, ...
            LTS_CORR_THRESH, DO_APPLY_CFO_CORRECTION, DO_APPLY_PHASE_ERR_CORRECTION, DO_APPLY_SFO_CORRECTION, ...
            MAX_TX_LEN, PHASE_CALIBRATION);
    avg_H = avg_H + H;
    
    subplot(3,1,1);
    plot(1:52, unwrap(angle(nonzeros(H(:, 1))))); hold on;
    plot(1:52, unwrap(angle(nonzeros(H(:, 2)))), 'g'); hold on;
    plot(1:52, unwrap(angle(nonzeros(H(:, 3)))), 'c'); hold on;
    plot(1:52, unwrap(angle(nonzeros(H(:, 4)))), 'r'); hold on;
    legend('RFA', 'RFB', 'RFC', 'RFD');
    ylabel('radian');
    title('phase respoonse');
    subplot(3,1,2);
    plot(1:52, unwrap(angle(nonzeros(H(:, 2)))) - unwrap(angle(nonzeros(H(:, 1)))), 'g'); hold on;
    plot(1:52, unwrap(angle(nonzeros(H(:, 3)))) - unwrap(angle(nonzeros(H(:, 1)))), 'c'); hold on;
    plot(1:52, unwrap(angle(nonzeros(H(:, 4)))) - unwrap(angle(nonzeros(H(:, 1)))), 'r'); hold on;
    legend('RFB-RFA', 'RFC-RFA', 'RFD-RFA');
    title('phase difference to RFA');
end   

INTER_RF_CALIBRATION = get_calibration_value(avg_H./ii);
save 'INTER_RF_CALIBRATION.mat' INTER_RF_CALIBRATION

% avg_rfa = avg_H(:,1)./ii;
% for ii=1:64
%     if avg_rfa(ii)==0
%         avg_rfa(ii)= 1;
%     else
%         avg_rfa(ii) = 1/avg_rfa(ii);
%     end
% end
% RFA_CALIBRATION = repmat(avg_rfa,1,4);
  
H = wl_example_siso_ofdm_txrx(preamble, nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ...
        ManualRxGainRF, ManualRxGainBB, RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL, ...
        USE_WARPLAB_TXRX, TX_SCALE, SC_IND_PILOTS, SC_IND_DATA, N_SC, CP_LEN, INTERP_RATE, FFT_OFFSET, ...
        LTS_CORR_THRESH, DO_APPLY_CFO_CORRECTION, DO_APPLY_PHASE_ERR_CORRECTION, DO_APPLY_SFO_CORRECTION, ...
        MAX_TX_LEN, PHASE_CALIBRATION);
calibrated_H = H .* INTER_RF_CALIBRATION; % .* RFA_CALIBRATION;
get_calibration_result(H, calibrated_H);
save calibrated_H.mat calibrated_H

subplot(3,1,3);
plot(1:52, unwrap(angle(nonzeros(calibrated_H(:, 1))))); hold on;
plot(1:52, unwrap(angle(nonzeros(calibrated_H(:, 2)))), 'g'); hold on;
plot(1:52, unwrap(angle(nonzeros(calibrated_H(:, 3)))), 'c'); hold on;
plot(1:52, unwrap(angle(nonzeros(calibrated_H(:, 4)))), 'r'); hold on;
legend('RFA', 'RFB', 'RFC', 'RFD');
ylabel('radian');
title('calibrated phase respoonse');
%% 2. TOF calibration for RFA
clear avg_H;
for ii=1:10
    PHASE_CALIBRATION = [0 0 0 0];
    H = wl_example_siso_ofdm_txrx(preamble, nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ...
            ManualRxGainRF, ManualRxGainBB, RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL, ...
            USE_WARPLAB_TXRX, TX_SCALE, SC_IND_PILOTS, SC_IND_DATA, N_SC, CP_LEN, INTERP_RATE, FFT_OFFSET, ...
            LTS_CORR_THRESH, DO_APPLY_CFO_CORRECTION, DO_APPLY_PHASE_ERR_CORRECTION, DO_APPLY_SFO_CORRECTION, ...
            MAX_TX_LEN, PHASE_CALIBRATION);
    
    avg_H(ii, :) = H(:,1).';
end

% tof_correction = exp(-1j.*angle(mean(avg_H).'));
tof_correction = exp(-1j.*2.*pi.*(7.13/50)./64.*[0:63].');
for ii=1:size(avg_H,1)
    H = avg_H(ii,:).';
    H_RFA = H(:,1);
    figure(3);  % clf;
    subplot(2,1,1);
    plot(unwrap(angle(nonzeros(H_RFA)))); hold on;
%     plot(unwrap(angle(fftshift(H(:,1))))); hold on;
    subplot(2,1,2);
    for tof=1:500*64
        cor(tof) = abs(sum(H_RFA.*exp(1j.*2.*pi.*(tof/500)./64.*[0:63].')))^2;
    end
    % plot(db(nonzeros(H(:,1))));
    plot(cor); hold on;
    [M, I] = max(cor);
%     ground_truth_distance = 2.366;
    fprintf('Measured tof %.2f ns\n', I*0.1);
    samp_delay = I*0.1/50;
%     fprintf('Measured tof %.2f ns, at %.3f meters away tof should be %.2f ns.\n', I*0.5, ground_truth_distance, ...
%         ground_truth_distance/3e8*1e9);
%     fprintf('Extra tof introduced by the RF chain: %.3f ns, wihch is %.3f sample delay.\n', ...
%         I*0.5 - ground_truth_distance/3e8*1e9, (I*0.5 - ground_truth_distance/3e8*1e9)/50);

    % RFA chain introduces ~0.989 extra sample delay (20 MHz sampling)
%     samp_delay = (I*0.5 - ground_truth_distance/3e8*1e9)/50;
    % tof_correction = exp(1j.*2.*pi.*(samp_delay)./64.*[0:63].');
    % tof_correction = exp(-1j.*angle(H(:,1)));
    

%     figure(4); % clf;
%     subplot(3,1,1);
% %     plot(unwrap(angle(nonzeros(H(:,1))))); hold on;
%     plot(unwrap(angle(H(:,1)))); hold on;
%     const_phase = exp(1j*angle(H(2,1)))/exp(-1j*2*pi*(samp_delay)/64*1);
%     plot(unwrap(angle(exp(-1j.*2.*pi.*(samp_delay)./64.*[0:63].').*const_phase)), 'r'); hold on;
%     subplot(3,1,2);
%     plot(unwrap(angle(H(:,1).*conj(exp(-1j.*2.*pi.*(samp_delay)./64.*[0:63].').*const_phase)))); hold on;
%     subplot(3,1,3);
%     plot(unwrap(angle(nonzeros(H(:,1).*tof_correction)))); hold on;
end

RFA_TOF_CALIBRATION = repmat(tof_correction, 1, 4);
save 'RFA_TOF_CALIBRATION.mat' RFA_TOF_CALIBRATION
%% collect experiment data

% H = wl_example_siso_ofdm_txrx(zeros(1,5000), nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ...
%         ManualRxGainRF, ManualRxGainBB, RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL, ...
%         USE_WARPLAB_TXRX, TX_SCALE, SC_IND_PILOTS, SC_IND_DATA, N_SC, CP_LEN, INTERP_RATE, FFT_OFFSET, ...
%         LTS_CORR_THRESH, DO_APPLY_CFO_CORRECTION, DO_APPLY_PHASE_ERR_CORRECTION, DO_APPLY_SFO_CORRECTION, ...
%         MAX_TX_LEN, PHASE_CALIBRATION);
% noise = get_per_tone_noise()


% RF_RX          = ifc_ids.RF_A;                    % Receive RF interface
% RF_RX_VEC      = [ifc_ids.RF_A];
% ManualRxGainRF = 2;
% ManualRxGainBB = 5;

% RF_RX          = ifc_ids.RF_B;                    % Receive RF interface
% RF_RX_VEC      = [ifc_ids.RF_B];
% ManualRxGainRF = 2;
% ManualRxGainBB = 5;

% RF_RX          = ifc_ids.RF_C;                    % Receive RF interface
% RF_RX_VEC      = [ifc_ids.RF_C];
% ManualRxGainRF = 2;
% ManualRxGainBB = 5;

% RF_RX          = ifc_ids.RF_D;                    % Receive RF interface
% RF_RX_VEC      = [ifc_ids.RF_D];
% ManualRxGainRF = 2;
% ManualRxGainBB = 5;

% ManualRxGainRF = [1 1 1 1]+1; %[3 3 3 3]                   % Rx RF Gain in [1:3] (ignored if USE_AGC is true)
% ManualRxGainBB = [16, 20, 16, 16]-15;
for ii =1:1
    PHASE_CALIBRATION = [0 0 0 0];
    H = wl_example_siso_ofdm_txrx(preamble, nodes, ifc_ids, USE_EXTERNAL_TRIGGER, USE_AGC, ...
            ManualRxGainRF, ManualRxGainBB, RF_TX, RF_RX, RF_RX_VEC, RF_TX_VEC, BAND, CHANNEL, ...
            USE_WARPLAB_TXRX, TX_SCALE, SC_IND_PILOTS, SC_IND_DATA, N_SC, CP_LEN, INTERP_RATE, FFT_OFFSET, ...
            LTS_CORR_THRESH, DO_APPLY_CFO_CORRECTION, DO_APPLY_PHASE_ERR_CORRECTION, DO_APPLY_SFO_CORRECTION, ...
            MAX_TX_LEN, PHASE_CALIBRATION);

%     calibrated_H = H .* INTER_RF_CALIBRATION(:, 1:size(H,2));
%     save(sprintf('./experiment_data/calibrated_H_%d.mat', ii), 'calibrated_H');
    % show_heatmap(H);
%     show_heatmap(calibrated_H);
%     pause(0.2);
end

% for ii=1:10
%     load(sprintf('./experiment_data/calibrated_H_%d.mat', ii));
%     show_heatmap(calibrated_H);
% end

% test_correction = exp(-1j.*2.*pi.*(0/50)./64.*[0:63].');
for ii=1:2
    H_RFA = H(:, ii); %calibrated_H(:,ii);
    figure(3);  % clf;
    subplot(2,1,1);
%     plot(unwrap(angle(nonzeros(H_RFA)))); hold on;
    plot(unwrap(angle(fftshift(H_RFA)))); hold on;
    subplot(2,1,2);
    tofs = -4*500:500*2;
    for jj=1:length(tofs)
        cor(jj) = abs(sum(H_RFA.*exp(1j.*2.*pi.*(tofs(jj)/500)./64.*[0:63].')))^2;
    end
    % plot(db(nonzeros(H(:,1))));
    plot(tofs.*0.1, cor); hold on;
    xlabel('ToF (ns)');
    [M, I] = max(cor);
%     ground_truth_distance = 2.366;
    fprintf('Measured tof %.2f ns\n', tofs(I)*0.1);
end

% HB = zeros(64,1);
% for ii=1:64
%     if H(ii)==0
%         HB(ii)= 1;
%     else
%         HB(ii) = 1/H(ii);
%     end
% end
% H.*HB

figure(7); 
plot(1:64, unwrap(angle(H.*[HA HB]))); hold on;


% figure;
% subplot(2,1,1);
% plot(unwrap(angle(nonzeros(H_A)))); hold on;
% plot(unwrap(angle(nonzeros(H_B)))); hold on;
% plot(unwrap(angle(nonzeros(H_C)))); hold on;
% plot(unwrap(angle(nonzeros(H_D)))); hold on;
% subplot(2,1,2);
% plot(unwrap(angle(nonzeros(H_B)./nonzeros(H_A)))); hold on;
% plot(unwrap(angle(nonzeros(H_C)./nonzeros(H_A)))); hold on;
% plot(unwrap(angle(nonzeros(H_D)./nonzeros(H_A)))); hold on;

% figure; 
% plot(angle(nonzeros(fftshift(H_A2))./nonzeros(fftshift(H_A1)))); hold on;
% plot(angle(nonzeros(fftshift(H_A3))./nonzeros(fftshift(H_A2))), 'r'); hold on;
% plot(angle(nonzeros(fftshift(H_A3))./nonzeros(fftshift(H_A2)))); hold on;
% plot(angle(nonzeros(fftshift(H_A1))), 'r'); hold on;
%%
    
load('INTER_RF_CALIBRATION.mat');
load('RFA_TOF_CALIBRATION.mat');
calibrated_H = H .* INTER_RF_CALIBRATION(:, 1:size(H,2)).* RFA_TOF_CALIBRATION(:, 1:size(H,2));
save calibrated_H.mat calibrated_H
save H.mat H

figure(4); clf; 
set(gcf, 'units', 'inches', 'position', [1 1 10 6]);
subplot(2,3,1);
plot(1:52, db(nonzeros(H(:, 1)))); hold on;
plot(1:52, db(nonzeros(H(:, 2))), 'g');
plot(1:52, db(nonzeros(H(:, 3))), 'c');
plot(1:52, db(nonzeros(H(:, 4))), 'r');
legend('RFA', 'RFB', 'RFC', 'RFD');
ylabel('dB');
title('uncalibrated mag respoonse');
subplot(2,3,4);
plot(1:52, unwrap(angle(nonzeros(H(:, 1))))); hold on;
plot(1:52, unwrap(angle(nonzeros(H(:, 2)))), 'g');
plot(1:52, unwrap(angle(nonzeros(H(:, 3)))), 'c');
plot(1:52, unwrap(angle(nonzeros(H(:, 4)))), 'r');
legend('RFA', 'RFB', 'RFC', 'RFD');
ylabel('radian');
title('uncalibrated phase respoonse');
subplot(2,3,2);
plot(1:52, db(nonzeros(calibrated_H(:, 1)))); hold on;
plot(1:52, db(nonzeros(calibrated_H(:, 2))), 'g');
plot(1:52, db(nonzeros(calibrated_H(:, 3))), 'c');
plot(1:52, db(nonzeros(calibrated_H(:, 4))), 'r');
legend('RFA', 'RFB', 'RFC', 'RFD');
ylabel('dB');
title('calibrated mag respoonse');
subplot(2,3,5);
plot(1:52, unwrap(angle(nonzeros(calibrated_H(:, 1))))); hold on;
plot(1:52, unwrap(angle(nonzeros(calibrated_H(:, 2)))), 'g');
plot(1:52, unwrap(angle(nonzeros(calibrated_H(:, 3)))), 'c');
plot(1:52, unwrap(angle(nonzeros(calibrated_H(:, 4)))), 'r');
legend('RFA', 'RFB', 'RFC', 'RFD');
ylabel('radian');
title('calibrated phase respoonse');
subplot(2,3,3);
% plot(1:52, db(nonzeros(calibrated_H(:, 1)))); hold on;
plot(1:52, db(nonzeros(INTER_RF_CALIBRATION(:, 2))), 'g'); hold on;
plot(1:52, db(nonzeros(INTER_RF_CALIBRATION(:, 3))), 'c');
plot(1:52, db(nonzeros(INTER_RF_CALIBRATION(:, 4))), 'r');
legend('RFB', 'RFC', 'RFD');
ylabel('dB');
title('mag calibration values');
subplot(2,3,6);
% plot(1:52, unwrap(angle(nonzeros(calibrated_H(:, 1))))); hold on;
plot(1:52, unwrap(angle(nonzeros(INTER_RF_CALIBRATION(:, 2)))), 'g'); hold on;
plot(1:52, unwrap(angle(nonzeros(INTER_RF_CALIBRATION(:, 3)))), 'c');
plot(1:52, unwrap(angle(nonzeros(INTER_RF_CALIBRATION(:, 4)))), 'r');
legend('RFB', 'RFC', 'RFD');
ylabel('radian');
title('phase calibration values');

show_heatmap(calibrated_H);
show_heatmap(H);
