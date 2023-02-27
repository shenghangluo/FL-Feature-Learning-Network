function [outputX_Data,outputY_Data,MSEX,MSEY] = LMS_2x2_PMDEq2(rx_signal_Filtered_X,rx_signal_Filtered_Y,...
                    pilotx,piloty, nr_tap_RRC, N_train, Power_scaling_rx)

rx_signal_Filtered_X = rx_signal_Filtered_X./Power_scaling_rx;
rx_signal_Filtered_Y = rx_signal_Filtered_Y./Power_scaling_rx;
pilotx = pilotx./rms(pilotx);
piloty = piloty./rms(pilotx);

 
Nf = 19;
k0 = floor(Nf/2);
mu_c = 0.001;
 
Index0_Rx_temp = (nr_tap_RRC-1)*2+1; % considering 2-times DSP overampling
 
reduced_inputX = rx_signal_Filtered_X(Index0_Rx_temp-k0:end);
reduced_inputY = rx_signal_Filtered_Y(Index0_Rx_temp-k0:end);
 
%~~~~~ C Matrices
C = zeros(2,2,Nf);
C(:,:,ceil(Nf/2)) = eye(2);
C11 = zeros(1,Nf);
C11(ceil(Nf/2)) = 1;
C12 = zeros(1,Nf);
C21 = zeros(1,Nf);
C22 = zeros(1,Nf);
C22(ceil(Nf/2)) = 1;

MSEX = zeros(1,length(rx_signal_Filtered_X(2*(nr_tap_RRC-1)+1:end-2*(nr_tap_RRC-1)))/2);
MSEY = zeros(1,length(rx_signal_Filtered_Y(2*(nr_tap_RRC-1)+1:end-2*(nr_tap_RRC-1)))/2);

outputX = zeros(1,length(rx_signal_Filtered_X(2*(nr_tap_RRC-1)+1:end-2*(nr_tap_RRC-1)))/2);
outputY = zeros(1,length(rx_signal_Filtered_Y(2*(nr_tap_RRC-1)+1:end-2*(nr_tap_RRC-1)))/2);
 
mse_counter = 0;
for k = Nf:2:2*(length(outputX))+Nf-1
    mse_counter = mse_counter+1;
    if ~mod(mse_counter,1e4)
        disp(['Counter = ' num2str(mse_counter) ' of ' num2str(length(outputX))])
    end
    
    s_combined = [reduced_inputX(k:-1:k-Nf+1).';reduced_inputY(k:-1:k-Nf+1).'];
    z1 = [C11 C12]*s_combined;
    z2 = [C21 C22]*s_combined;
    
    if k<= 2*(N_train+k0)
        
        e1 = z1-pilotx(mse_counter);
        e2 = z2-piloty(mse_counter);
        
        
        for eq_tap = 1:Nf
            
            C(:,:,eq_tap) = [C11(eq_tap) C12(eq_tap);...
                C21(eq_tap) C22(eq_tap)];
            C(:,:,eq_tap) = C(:,:,eq_tap)-...
                mu_c*[e1;e2]*[reduced_inputX(k-eq_tap+1);...
                reduced_inputY(k-eq_tap+1)]';
            
            C11(eq_tap) = C(1,1,eq_tap);
            C12(eq_tap) = C(1,2,eq_tap);
            C21(eq_tap) = C(2,1,eq_tap);
            C22(eq_tap) = C(2,2,eq_tap);
            
        end
        MSEX(mse_counter) = (abs(e1).^2);
        MSEY(mse_counter) = (abs(e2).^2);
    end
    
    outputX(mse_counter) = z1;
    outputY(mse_counter) = z2;
    
end
 
outputX_Data = outputX(N_train+1:end).*Power_scaling_rx;
outputY_Data = outputY(N_train+1:end).*Power_scaling_rx;
%outputX = outputX.*Power_scaling_rx;
%outputY = outputY.*Power_scaling_rx;

return
