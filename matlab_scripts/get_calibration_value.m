function [rf_calibration] = get_calibration_value(H)
%     % version 1: estimating a constant phase difference between RF chains
%     rfa_phase = unwrap(angle(nonzeros(H(:,1))));
%     rfb_phase = unwrap(angle(nonzeros(H(:,2))));
%     rfc_phase = unwrap(angle(nonzeros(H(:,3))));
%     rfd_phase = unwrap(angle(nonzeros(H(:,4))));
%     
%     diff_a_b = rfa_phase - rfb_phase;
%     diff_a_c = rfa_phase - rfc_phase;
%     diff_a_d = rfa_phase - rfd_phase;
% %     [diff_a_b diff_a_c diff_a_d]
%     
%     idxs1 = abs(diff_a_b)>2*pi;
%     idxs2 = abs(diff_a_c)>2*pi;
%     idxs3 = abs(diff_a_d)>2*pi;
%     
%     diff_a_b(idxs1) = mod(diff_a_b(idxs1), sign(diff_a_b(idxs1))*2*pi);
%     diff_a_c(idxs2) = mod(diff_a_c(idxs2), sign(diff_a_c(idxs2))*2*pi);
%     diff_a_d(idxs3) = mod(diff_a_d(idxs3), sign(diff_a_d(idxs3))*2*pi);
%     
%     rf_calibration = mean([diff_a_b diff_a_c diff_a_d]);


    % version 2: estimating phase difference with a slope
    zero_idx = find(H(:,1)==0);
    H_copy = H;
    H_copy(zero_idx, 2:4) = ones(length(zero_idx), 3);
    rf_calibration = [ones(size(H,1), 1) H_copy(:,1)./H_copy(:,2) H_copy(:,1)./H_copy(:,3) H_copy(:,1)./H_copy(:,4)];
end