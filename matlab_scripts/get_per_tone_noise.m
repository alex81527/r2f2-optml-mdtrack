function [noise] = get_per_tone_noise()
    load('rfa_noise');
    load('rfb_noise');
    load('rfc_noise');
    load('rfd_noise');

    noise = [mean(abs(rfa_noise).^2) mean(abs(rfb_noise).^2) mean(abs(rfc_noise).^2) mean(abs(rfd_noise).^2)];
    
%     ffta = fft(rfa_noise(2001:2064), 64);
%     fftb = fft(rfb_noise(2001:2064), 64);
%     fftc = fft(rfc_noise(2001:2064), 64);
%     fftd = fft(rfd_noise(2001:2064), 64);
% 
%     figure(66);
%     plot(db(ffta));
%     plot(db(fftb), 'r');
%     plot(db(fftc), 'g');
%     plot(db(fftd), 'c');
end