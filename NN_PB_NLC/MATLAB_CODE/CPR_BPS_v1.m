function [E_out] = CPR_BPS_v1(E_in, I1, N1, M)


% zero padding
rx_lmd_x=E_in;

symbol_num=length(rx_lmd_x);
rx_lmd_x=reshape(rx_lmd_x,symbol_num,1); % to ensure to be column vectors

padding = zeros(floor(N1)/2,1);
rx = [padding;rx_lmd_x;padding];

% Set up testing phase

for jj = 0:I1-1
    p_t(jj+1) = jj*pi/(2*I1)-pi/4;
end


% Phase estimation


rx_new=rx*exp(1j*p_t);  %%%% transform into matrix%%%

if M==16
    rx_dec_new=Dec_16QAM(rx_new);
    elseif  M==4
    rx_dec_new=Dec_4QAM(rx_new);    
    elseif  M==8
    rx_dec_new=Dec_8QAM(rx_new);    
        elseif  M==32
    rx_dec_new=Dec_32QAM(rx_new);    
            elseif  M==64
    rx_dec_new=Dec_64QAM(rx_new);    
end


for i = 1:length(rx_lmd_x)
    
     M1(i,1:length(p_t))=sum((abs(rx_new(i:N1+i-1,1:length(p_t)) - rx_dec_new(i:N1+i-1,1:length(p_t))).^2));
             
     [m,p_min] = min(M1(i,1:length(p_t)));
    
     p_dec(i) = (p_min - 1)*pi/(2*I1) - pi/4;
end



%% Phase unwrapping

phase_x_unwrapped = zeros(1,length(p_dec));

phase_x_unwrapped(1) = p_dec(1);

fx = zeros(1,length(p_dec)-1);
cx = 0;

for ii = 2:length(p_dec)
    if abs(p_dec(ii) - p_dec(ii - 1)) <= pi/4;
        fx(ii) = 0;
    elseif p_dec(ii) - p_dec(ii - 1) < -pi/4;
        fx(ii) = 1;
    else fx(ii) = -1;
    end
    cx = cx + (pi/2)*fx(ii);
    phase_x_unwrapped(ii) = p_dec(ii) + cx;
end

E_out = rx_lmd_x.*exp(1i*phase_x_unwrapped.');

% figure;plot(phase_x_unwrapped);

end
