% Filtra en racine de cosinus surélevé (RRC filter)
% J. Wang JLT-30, n°23, pp.3679-3686, Dec. 2012, 

function f = RRC_filter(Ax, Bbaud, beta, Bo)

totalsamples = length(Ax);
                  
f = (-totalsamples/2+1:totalsamples/2)/totalsamples;
f = f * Bo;
w = 2*pi.*f;


% disp(sprintf('frequency min = %g GHz', (1-beta)*Bbaud/2 * 1e-9))
% disp(sprintf('frequency max = %g GHz', (1+beta)*Bbaud/2 * 1e-9))

[a,b] = find(f >= - (1-beta)*Bbaud/2);   f_low_negative = b(1,1);
[a,b] = find(f >= - (1+beta)*Bbaud/2);   f_high_negative = b(1,1);
[a,b] = find(f >= (1-beta)*Bbaud/2);     f_low_positive = b(1,1);
[a,b] = find(f >= (1+beta)*Bbaud/2);     f_high_positive = b(1,1);



H_RC = size(1, totalsamples);
H_RC = 1 + cos(pi/(beta*Bbaud) .* (abs(f) - (1-beta)*Bbaud/2));   H_RC = H_RC/2;
H_RC(1,1:f_high_negative) = 0.0;  H_RC(1, f_high_positive:end) = 0.0;
H_RC(1, f_low_negative:f_low_positive) = 1.0;

H_RRC = sqrt(H_RC);


% figure(55)
% hold on
% plot(f*1e-9, H_RC)
% plot(f*1e-9, H_RRC,'r')
% xlabel('Frequency in GHz')
% hold off


f= ifft(fft(Ax) .* fftshift(H_RRC));