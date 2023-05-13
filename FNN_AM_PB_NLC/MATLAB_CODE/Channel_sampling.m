function [Tx_Symbol_Channel]= Channel_sampling(Tx_Symbol_RRC_Filt, Channel_upsamling_factor, No_sub_channels, RRC_beta, FP_xi, RRC_tau, Total_upsamling_factor, RRC_No_of_taps)

Tx_Symbol_Oversampled = zeros(No_sub_channels, length(Tx_Symbol_RRC_Filt)*Channel_upsamling_factor);
for ch_loop = 1:No_sub_channels
    Tx_Symbol_Oversampled(ch_loop,:) = interp(Tx_Symbol_RRC_Filt(ch_loop,:),Channel_upsamling_factor);
end

figure
        my_plot_fft(Tx_Symbol_Oversampled(1,:), Total_upsamling_factor, 64e9, 'norm')
        
Range = 1:size(Tx_Symbol_Oversampled,2); % time index
subchannel_indices = (0:No_sub_channels-1)-(No_sub_channels-1)/2; % frequency index
freq_exp_factor = exp(1i*2*pi*(1+RRC_beta)*FP_xi*RRC_tau/Total_upsamling_factor*...
     repmat(Range, No_sub_channels,1).*repmat(subchannel_indices.',1,size(Tx_Symbol_Oversampled,2))); %frequency shift

Tx_Symbol_Summed = sum(Tx_Symbol_Oversampled.*freq_exp_factor,1); % freq shift and combine

Tx_Symbol_Scaling = sqrt(No_sub_channels);
Tx_Symbol_Channel = Tx_Symbol_Summed./Tx_Symbol_Scaling; % make average transmit power per sample is 1

figure
        my_plot_fft(Tx_Symbol_Channel(1,:), Total_upsamling_factor, 64e9, 'norm')
return
    
    