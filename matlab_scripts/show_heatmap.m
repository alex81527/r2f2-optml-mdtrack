function [heatmap] = show_heatmap(H)
    tof_step = 5e-9;    
    tofs = -10:tof_step/50e-9:10;
    coss = 1:-0.05:-1;
    for ii = 1:length(coss)
        for jj = 1:length(tofs)
            tof_mat = repmat(exp(1j.*2.*pi.*(tofs(jj))./64.*[0:63].'),1,size(H,2));
            aoa_mat = repmat(exp(1j.*2.*pi.*0.5.*coss(ii).*[0:size(H,2)-1]), size(H,1), 1);
            heatmap(ii, jj) = abs(sum(sum(H.*tof_mat.*aoa_mat)))^2;
        end
    end
    
    peak = max(abs(heatmap(:)));
    [row, col] = find(abs(heatmap)==peak);
    fprintf('peak at %.3f deg %.3f ns\n', rad2deg(acos(coss(row))), tofs(col)*50);
    
    figure(678); clf;
    imagesc(heatmap./max(max(abs(heatmap))));
    xlabel('Tof (ns)');
    ylabel('AoA (deg)');
    set(gca, 'XTick', [1: floor(length(tofs)/10):length(tofs)]);
    set(gca, 'xticklabel', [50*tofs(1: floor(length(tofs)/10) :length(tofs))]);
    set(gca, 'YTick', [1: floor(length(coss)/10):length(coss)]);
    set(gca, 'yticklabel', [rad2deg(acos(coss(1: floor(length(coss)/10) :length(coss))))]);
    colormap jet;
    colorbar;

    figure(679);
    scatter(tofs(col)*50, rad2deg(acos(coss(row))), 25); hold on;
    axis([tofs(1)*50 tofs(length(tofs))*50 rad2deg(acos(coss(1))) rad2deg(acos(coss(length(coss))))]);
    xlabel('Tof (ns)');
    ylabel('AoA (deg)');
    
end