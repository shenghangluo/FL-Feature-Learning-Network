function [Tx_Symbol_Channel] = Channel_sampling_new(Tx_Symbol_RRC_Filt, Channel_upsamling_factor, Total_Sub_Channels, BaudRate, WDM_Channel_Spacing, Ch_index, Total_upsamling_factor)

    Tx_Symbol_Oversampled = zeros(Total_Sub_Channels, length(Tx_Symbol_RRC_Filt)*Channel_upsamling_factor);
    for ch_loop = 1:Total_Sub_Channels
        Tx_Symbol_Oversampled(ch_loop,:) = interp(Tx_Symbol_RRC_Filt(ch_loop,:),Channel_upsamling_factor);
    end

    Range = 1:size(Tx_Symbol_Oversampled,2); % time index
    freq_exp_factor = exp(1i*2*pi*(WDM_Channel_Spacing/BaudRate)/Total_upsamling_factor*...
         repmat(Range, Total_Sub_Channels,1).*repmat(Ch_index.',1,size(Tx_Symbol_Oversampled,2))); %frequency shift

    Tx_Symbol_Summed = sum(Tx_Symbol_Oversampled.*freq_exp_factor,1); % freq shift and combine

    Tx_Symbol_Scaling = sqrt(Total_Sub_Channels);
    Tx_Symbol_Channel = Tx_Symbol_Summed./Tx_Symbol_Scaling; % make average transmit power per sample is 1
return
    
    