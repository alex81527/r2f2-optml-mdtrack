function [avg_phase_diff] = get_calibration_result(calibrated_H)
    p = angle(calibrated_H);
    p = p(:,2:4) - repmat(p(:,1), 1, size(p, 2)-1);
    for m = 1:size(p, 1)
        for n = 1:size(p,2)
            if p(m,n) > 0 && abs(mod(p(m,n),-2*pi)) < abs(p(m,n))
                p(m,n) = mod(p(m,n),-2*pi);
            elseif p(m,n) < 0 && abs(mod(p(m,n),2*pi)) < abs(p(m,n))
                p(m,n) = mod(p(m,n),2*pi);
            end
        end
    end
    avg_phase_diff = mean(abs(p))
end