function Phi_opt = DBP_Optimization(Rx_SymbolsXY, BaudRate, Digital_upsampling_factor,...
                                BP_Nslices, lambda0, Chrom_Disp, Gama, loss_dB, SpanLength,...
                                No_of_span, rrc_output_rx, RRC_No_of_taps, rrc2, Power_scaling_rx, ...
                                Tx_BitsX, Tx_BitsY, Tx_SymbolsX, Tx_SymbolsY)
    Phi = 0.45:0.01:0.85;
    MMSE = zeros(size(Phi));
    
    Rx_X = Rx_SymbolsXY(1,:);
    Rx_Y = Rx_SymbolsXY(2,:);
%     Rx_X_in=Rx_X(1, (RRC_No_of_taps-1)/1+1:end-(RRC_No_of_taps-1)/1);   %2048
%     Rx_Y_in=Rx_Y(1, (RRC_No_of_taps-1)/1+1:end-(RRC_No_of_taps-1)/1);
    Rx_SymbolsXY_in = [Rx_X; Rx_Y];
    for phi = 1:1:length(Phi)
        BPoutput_xy = BackPropagation_VECTORIAL1(Rx_SymbolsXY_in,BaudRate,Digital_upsampling_factor,...
                    Phi(phi), BP_Nslices, lambda0, Chrom_Disp, Gama, ...
                                loss_dB, SpanLength, No_of_span);
        Rx_SymbolsX_CDCorBP = BPoutput_xy(1,:);
        Rx_SymbolsY_CDCorBP = BPoutput_xy(2,:);



        %% Receiver RRC filter (invers tx RRC filter)
        Rx_SymbolsX_Filt =  conv(Rx_SymbolsX_CDCorBP, rrc_output_rx);   % Receiver filtering for baseband center channel
        Rx_SymbolsY_Filt =  conv(Rx_SymbolsY_CDCorBP, rrc_output_rx);
        
        Rx_SymbolsX_Temp_ds_x1 = downsample(Rx_SymbolsX_Filt, Digital_upsampling_factor);
        Rx_SymbolsY_Temp_ds_x1 = downsample(Rx_SymbolsY_Filt, Digital_upsampling_factor);
        
        Rx_SymbolsX_Filt_norm = Rx_SymbolsX_Temp_ds_x1./sqrt(Digital_upsampling_factor);
        Rx_SymbolsY_Filt_norm = Rx_SymbolsY_Temp_ds_x1./sqrt(Digital_upsampling_factor);
        
       Rx_SymbolsX_Temp = Rx_SymbolsX_Filt_norm((RRC_No_of_taps+rrc2-2)/2+1:end-(RRC_No_of_taps+rrc2-2)/2);
        Rx_SymbolsY_Temp = Rx_SymbolsY_Filt_norm((RRC_No_of_taps+rrc2-2)/2+1:end-(RRC_No_of_taps+rrc2-2)/2);

        Rx_SymbolsX_Scaled = Rx_SymbolsX_Temp./Power_scaling_rx; % Power scaling to make it unit
        Rx_SymbolsY_Scaled = Rx_SymbolsY_Temp./Power_scaling_rx; % rms(Rx_SymbolsY);

	%%% Phase Rotation
	sx= xcorr(Tx_SymbolsX(1, :), Rx_SymbolsX_Scaled(1, 1:end));
        % Here no pilots used for the estmation of phase rotation. The actual tx signals are used to estimate the phase rotation
        sx_max = max(sx);
        phi_x = angle (sx_max);
        Rx_SymbolsX_Com = Rx_SymbolsX_Scaled(1, 1:end).*exp(1i*phi_x);
            
        sy= xcorr(Tx_SymbolsY(1, :), Rx_SymbolsY_Scaled(1,1:end));
        sy_max = max(sy);
        phi_y = angle (sy_max);
        Rx_SymbolsY_Com = Rx_SymbolsY_Scaled(1, 1:end).*exp(1i*phi_y);
        %% BER calculations
M=64;
        Rx_BitsX=transpose(qamdemod(Rx_SymbolsX_Com.', M,'gray',  'OutputType', 'bit', 'UnitAveragePower', true));%, 'NoiseVariance', Norm_Noise_PowerX_Mes); %M QAM soft demod
        Rx_BitsY=transpose(qamdemod(Rx_SymbolsY_Com.', M, 'gray', 'OutputType', 'bit', 'UnitAveragePower', true));%, 'NoiseVariance', Norm_Noise_PowerY_Mes);

        Frame_ErrorsX = sum(Tx_BitsX(1, 101:end-100)~=Rx_BitsX(1, 101:end-100));  % Errors in a loop (x code blocks)
        Frame_ErrorsY = sum(Tx_BitsY(1, 101:end-100)~=Rx_BitsY(1, 101:end-100));
%         Frame_ErrorsX = sum(Tx_BitsX(COI_no, 1:100)~=Rx_BitsX(1:100));  % Errors in a loop (x code blocks)
%         Frame_ErrorsY = sum(Tx_BitsY(COI_no, 1:100)~=Rx_BitsY(1:100));

%         Total_Errors = Total_Errors+Frame_ErrorsX+Frame_ErrorsY;
%         Total_Bits = Total_Bits+2*length(Rx_BitsX)-400;

%% Only considering error from X-pol
        Total_Errors = Frame_ErrorsX;
        Total_Bits = length(Rx_BitsX)-400;

   BER(phi) = Total_Errors/Total_Bits;
%    Q_Square_factor(phi)=20*log10(sqrt(10)*erfcinv((8/3)*BER(phi)));
   Q(phi) = 20*log10(sqrt(2)*erfcinv(2*BER(phi)));            
    end
    [min_val, min_ind] = min(BER);
    Phi_opt = Phi(min_ind)
%     Q  = 20*log10(sqrt(10)*erfcinv((8/3)*min_val))
    Q = 20*log10(sqrt(2)*erfcinv(2*min_val))
    hold on