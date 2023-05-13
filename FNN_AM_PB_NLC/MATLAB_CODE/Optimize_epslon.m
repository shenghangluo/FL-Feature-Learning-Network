function epslone_opt = Optimize_epslon(Tx_SymbolsX_Frame, Tx_SymbolsY_Frame, Tx_Pow_dBm, Digital_upsampling_factor,...
                                rrc_output_tx, Channel_upsamling_factor, No_sub_channels, RRC_beta, FP_xi,...
                                RRC_tau, Total_upsamling_factor, RRC_No_of_taps, BaudRate, Channel_Spacing, ...
                            lambda0, Chrom_Disp, Gama, loss_dB, Noise_Figure_dB, SpanLength, ...
                            No_of_span, SSFM_slices_per_span, AWGN, PMD_Flag, PMD, rrc_output_rx,rrc2, Head_Tail_Len, ...
                            Pilot_Len)

Power_scaling_factor=1.0:0.4:3;
MMSE = zeros(size(Power_scaling_factor));

COI_no = (No_sub_channels+1)/2;
Precomp_Tx_SymbolsX_Frame = Tx_SymbolsX_Frame;
Precomp_Tx_SymbolsY_Frame = Tx_SymbolsY_Frame;

for loop1 = 1:1:length(Power_scaling_factor)
    epslon = Power_scaling_factor(loop1);
       
%     delta_x_first_order=perturb_first_order_LOW_COMPLEXITY_NEW(Tx_SymbolsX_Frame(COI_no, :), Tx_SymbolsY_Frame(COI_no, :), Tx_Pow_dBm, epslon);
%     delta_y_first_order=perturb_first_order_LOW_COMPLEXITY_NEW(Tx_SymbolsY_Frame(COI_no, :), Tx_SymbolsX_Frame(COI_no, :), Tx_Pow_dBm, epslon);
    Precomp_Tx_SymbolsX_Frame(COI_no, :) = Tx_SymbolsX_Frame(COI_no, :);%-delta_x_first_order;
    Precomp_Tx_SymbolsY_Frame(COI_no, :) = Tx_SymbolsY_Frame(COI_no, :);%-delta_y_first_order;

    
    %% DSP up sampling by 2
    Tx_SymbolsX_Dig_upsamples = transpose(upsample(transpose(Precomp_Tx_SymbolsX_Frame),Digital_upsampling_factor)); % Upsampling by 2
    Tx_SymbolsY_Dig_upsamples = transpose(upsample(transpose(Precomp_Tx_SymbolsY_Frame),Digital_upsampling_factor));

    %% To keep the power normalized
    Tx_SymbolsX_Dig_upsamples_norm = sqrt(Digital_upsampling_factor).*Tx_SymbolsX_Dig_upsamples;   
    Tx_SymbolsY_Dig_upsamples_norm = sqrt(Digital_upsampling_factor).*Tx_SymbolsY_Dig_upsamples;

    %% RRC filtering
    Tx_SymbolX_RRC_Filt = conv2(Tx_SymbolsX_Dig_upsamples_norm, rrc_output_tx); % rrc pulse filtering
    Tx_SymbolY_RRC_Filt = conv2(Tx_SymbolsY_Dig_upsamples_norm, rrc_output_tx);

    %% CD compensation
        CDC_flag=(Chrom_Disp~=0);
        [Tx_SymbolX_RRC_Filt, Tx_SymbolY_RRC_Filt] = CD_Compensation_Lumped(CDC_flag, Tx_SymbolX_RRC_Filt, Tx_SymbolY_RRC_Filt, 0, Digital_upsampling_factor*BaudRate, No_of_span*SpanLength*10^3, 0.5*Chrom_Disp, lambda0);
 
    %% Channel select and sum the signals
    Tx_SymbolX_Channel = Channel_sampling(Tx_SymbolX_RRC_Filt, Channel_upsamling_factor, No_sub_channels, RRC_beta, FP_xi, RRC_tau, Total_upsamling_factor, RRC_No_of_taps); % Upsample, frequency shift and combine
    Tx_SymbolY_Channel = Channel_sampling(Tx_SymbolY_RRC_Filt, Channel_upsamling_factor, No_sub_channels, RRC_beta, FP_xi, RRC_tau, Total_upsamling_factor, RRC_No_of_taps);
       
        
    %% Fix the transmit power
    Power_scaling = sqrt((10^((Tx_Pow_dBm-30)/10)*(No_sub_channels)/2)); % /2 comes due to the divition into 2 polarizations
    Power_scaling_rx = sqrt((10^((Tx_Pow_dBm-30)/10))/2);

    Tx_Symbols_Channel = (Power_scaling).*[Tx_SymbolX_Channel; Tx_SymbolY_Channel]; % scaled accourding to the tx power
        
   

    %% Send via fiber channel SSFM        
    %{%
    Rx_Symbols_Channel = FiberTransmissionLink_SSFM(Tx_Symbols_Channel,BaudRate, ...
                            Total_upsamling_factor, lambda0, Chrom_Disp, Gama, ...
                            loss_dB, Noise_Figure_dB, SpanLength, No_of_span, SSFM_slices_per_span, ...
                            AWGN, PMD_Flag, PMD);

    %% Channel selection
    COI_no = (No_sub_channels+1)/2;
    cf_m = ((0:No_sub_channels-1)-(No_sub_channels-1)/2)*Channel_Spacing;
    cf = cf_m(COI_no);
    tt = (1:size(Rx_Symbols_Channel, 2))/(Total_upsamling_factor*BaudRate);
    Rx_SymbolsX_Channel = Rx_Symbols_Channel(1,:) .* exp(-1i*2*pi*cf*tt);
    Rx_SymbolsY_Channel = Rx_Symbols_Channel(2,:) .* exp(-1i*2*pi*cf*tt);
        
    %% Optical Filtering for center channel
    totalsamples = length(Rx_SymbolsX_Channel);
    f = (-totalsamples/2+1:totalsamples/2)/totalsamples;
    f = fftshift(f) * Total_upsamling_factor*BaudRate; 
    mf = 20;  Bopt = Channel_Spacing/2; %((1+RRC_beta)*BaudRate)/2;  
    Hopt = exp(-(f/Bopt).^(2*mf));
    Rx_SymbolsX_Opt_filt  = ifft(Hopt .* fft(Rx_SymbolsX_Channel));
    Rx_SymbolsY_Opt_filt  = ifft(Hopt .* fft(Rx_SymbolsY_Channel));
        
    %% Down sample to x2 sample domain
    Rx_SymbolsX_downsamp_x2=downsample(Rx_SymbolsX_Opt_filt, Channel_upsamling_factor); % Downsampling
    Rx_SymbolsY_downsamp_x2=downsample(Rx_SymbolsY_Opt_filt, Channel_upsamling_factor);
        

%% CD compensation
    CDC_flag=(Chrom_Disp~=0);
    [Rx_SymbolsX_CDC, Rx_SymbolsY_CDC] = CD_Compensation_Lumped(CDC_flag, Rx_SymbolsX_downsamp_x2, Rx_SymbolsY_downsamp_x2, cf, Digital_upsampling_factor*BaudRate, No_of_span*SpanLength*10^3, 0.5*Chrom_Disp, lambda0);

        %% Receiver RRC filter (invers tx RRC filter)
        Rx_SymbolsX_Filt =  conv(Rx_SymbolsX_CDC, rrc_output_rx);   % Receiver filtering for baseband center channel
        Rx_SymbolsY_Filt =  conv(Rx_SymbolsY_CDC, rrc_output_rx);

        Rx_SymbolsX_Filt_norm = Rx_SymbolsX_Filt./sqrt(Digital_upsampling_factor);
        Rx_SymbolsY_Filt_norm = Rx_SymbolsY_Filt./sqrt(Digital_upsampling_factor);

%% Downsampled to symbol domain
        Rx_SymbolsX_Temp_ds_x1 = downsample(Rx_SymbolsX_Filt_norm, Digital_upsampling_factor);
        Rx_SymbolsY_Temp_ds_x1 = downsample(Rx_SymbolsY_Filt_norm, Digital_upsampling_factor);

        %% Ignore the filter residual at head and tail
        Rx_SymbolsX_Temp = Rx_SymbolsX_Temp_ds_x1((RRC_No_of_taps+rrc2-2)/2+1:end-(RRC_No_of_taps+rrc2-2)/2);
        Rx_SymbolsY_Temp = Rx_SymbolsY_Temp_ds_x1((RRC_No_of_taps+rrc2-2)/2+1:end-(RRC_No_of_taps+rrc2-2)/2);
        
        %% Scale back the power to 1 of received symbols
        Rx_SymbolsX_Temp_Sc = Rx_SymbolsX_Temp./Power_scaling_rx;
        Rx_SymbolsY_Temp_Sc = Rx_SymbolsY_Temp./Power_scaling_rx;
        
        %% Phase recovery and ignore the redundant symbols transmitted at head and tail
        Phase_com_flag = 1;
        if(Phase_com_flag)
            sx= xcorr(Tx_SymbolsX(COI_no, :), Rx_SymbolsX_Temp_Sc(Head_Tail_Len+Pilot_Len+1:end-Head_Tail_Len));
            % Here no pilots used for the estmation of phase rotation. The actual tx signals are used to estimate the phase rotation
            sx_max = max(sx);
            phi_x = angle (sx_max);
            Rx_SymbolsX_Com = Rx_SymbolsX_Temp_Sc(Head_Tail_Len+Pilot_Len+1:end-Head_Tail_Len).*exp(1i*phi_x);

            sy= xcorr(Tx_SymbolsY(COI_no, :), Rx_SymbolsY_Temp_Sc(Head_Tail_Len+Pilot_Len+1:end-Head_Tail_Len));
            sy_max = max(sy);
            phi_y = angle (sy_max);
            Rx_SymbolsY_Com = Rx_SymbolsY_Temp_Sc(Head_Tail_Len+Pilot_Len+1:end-Head_Tail_Len).*exp(1i*phi_y);
        else
            Rx_SymbolsX_Com = Rx_SymbolsX_Temp_Sc(Head_Tail_Len+Pilot_Len+1:end-Head_Tail_Len);
            Rx_SymbolsY_Com = Rx_SymbolsY_Temp_Sc(Head_Tail_Len+Pilot_Len+1:end-Head_Tail_Len);
        end

        
                %% QAM Demapper
        Rx_BitsX=transpose(qamdemod(Rx_SymbolsX_Com.', M,  'OutputType', 'bit', 'UnitAveragePower', true));%, 'NoiseVariance', Norm_Noise_PowerX_Mes); %M QAM soft demod
        Rx_BitsY=transpose(qamdemod(Rx_SymbolsY_Com.', M,  'OutputType', 'bit', 'UnitAveragePower', true));%, 'NoiseVariance', Norm_Noise_PowerY_Mes);

        %% BER calculations

        Frame_ErrorsX = sum(Tx_BitsX(COI_no, :)~=Rx_BitsX);  % Errors in a loop (x code blocks)
        Frame_ErrorsY = sum(Tx_BitsY(COI_no, :)~=Rx_BitsY);
        
%% For only X-polarization BER calculation

        Total_Errors = Total_Errors+Frame_ErrorsX;
        Total_Bits = Total_Bits+length(Rx_BitsX)-400;
              
        loop_count = loop_count+1; % just to control loop during testing
 
   BER(loop1, pow_loop) = Total_Errors/Total_Bits
   Q(loop1, pow_loop)=20*log10(sqrt(2)*erfcinv(2*BER(1, pow_loop)))
   
end

[min_val, min_ind] = min(BER);
epslone_opt = Power_scaling_factor(min_ind);
    
end