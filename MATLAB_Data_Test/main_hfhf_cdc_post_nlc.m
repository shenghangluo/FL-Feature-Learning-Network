clear all
close all
clc


Tx_Pow_dBm = 2;
    config;


    errors_cdc = zeros(No_Sub_Channels, NoLoops);
    bit_count_cdc = zeros(No_Sub_Channels, NoLoops);
    ber_cdc = zeros(No_Sub_Channels, NoLoops);
    q_factor_db_cdc = zeros(No_Sub_Channels, NoLoops);

    

    for loop1 = 1:1:NoLoops
        disp(['Running for Tx power: ', num2str(Tx_Pow_dBm), ' dBm; Loop:- ', num2str(loop1), '/', num2str(NoLoops)]);
        rng(Data_rand_seed+loop1*1234+4567);

        % Tx bits and tx QAM symbol data frame generation
        Tx_BitsX = randi([0,1], Total_Sub_Channels, Frame_Len*log2(M)); % Random bit generation
        Tx_BitsY = randi([0,1], Total_Sub_Channels, Frame_Len*log2(M)); % Random bit generation

        Tx_SymbolsX = transpose(qammod(Tx_BitsX.', M,  'InputType', 'bit', 'UnitAveragePower', true));  %M QAM modulation 
        Tx_SymbolsY = transpose(qammod(Tx_BitsY.', M,  'InputType', 'bit', 'UnitAveragePower', true));  %M QAM modulation       

        %% ---------------------------------Please activate this section if you want to generate new information bits----------------------- %% 
        save ('Tx_BitsX.txt', 'Tx_BitsX','-ascii');
        save ('Tx_BitsY.txt', 'Tx_BitsY','-ascii');
%% ---------------------------------------------------------------------------------------------------------------------------------- %% 

%         load ('Tx_BitsX.txt', 'Tx_BitsX','-ascii');
%         load ('Tx_BitsY.txt', 'Tx_BitsY','-ascii');
%         Tx_SymbolsX(1, :) = transpose(qammod(Tx_BitsX(1, :).', M,  'InputType', 'bit', 'UnitAveragePower', true));  %M QAM modulation 
%         Tx_SymbolsY(1, :) = transpose(qammod(Tx_BitsY(1, :).', M,  'InputType', 'bit', 'UnitAveragePower', true));   %M QAM modulation 
%% ----------------------------------------------------------------------------------------------------------------------------------- %%
        if(Head_Tail_Len>0)% Head and tail of frame to mitigate ssfm residual effects
            Head_SymbolsX = qammod(randi([0,M-1], Total_Sub_Channels, Head_Tail_Len), M,  'InputType', 'integer', 'UnitAveragePower', true);
            Tail_symbolsX = qammod(randi([0,M-1], Total_Sub_Channels, Head_Tail_Len), M,  'InputType', 'integer', 'UnitAveragePower', true);
            Head_SymbolsY = qammod(randi([0,M-1], Total_Sub_Channels, Head_Tail_Len), M,  'InputType', 'integer', 'UnitAveragePower', true);
            Tail_symbolsY = qammod(randi([0,M-1], Total_Sub_Channels, Head_Tail_Len), M,  'InputType', 'integer', 'UnitAveragePower', true);
        else
            Head_SymbolsX = [];
            Tail_symbolsX = [];
            Head_SymbolsY = [];
            Tail_symbolsY = [];
        end

        if(CPC_Pilot_len>0)% Pilot symbols to perfrom constant phase correction (CPC)
            CPC_Pilot_SymbolsX = qammod(randi([0,M-1], Total_Sub_Channels, CPC_Pilot_len), M,  'InputType', 'integer', 'UnitAveragePower', true);
            CPC_Pilot_SymbolsY = qammod(randi([0,M-1], Total_Sub_Channels, CPC_Pilot_len), M,  'InputType', 'integer', 'UnitAveragePower', true);
        else
            CPC_Pilot_SymbolsX = [];
            CPC_Pilot_SymbolsY = [];
        end

        if(Pilot_Len>0)% Pilot symbols to train LMSE
            Pilot_SymbolsX = qammod(randi([0,M-1], Total_Sub_Channels, Pilot_Len), M,  'InputType', 'integer', 'UnitAveragePower', true);
            Pilot_SymbolsY = qammod(randi([0,M-1], Total_Sub_Channels, Pilot_Len), M,  'InputType', 'integer', 'UnitAveragePower', true);
        else
            Pilot_SymbolsX = [];
            Pilot_SymbolsY = [];
        end

        Tx_SymbolsX_Frame = [Head_SymbolsX, CPC_Pilot_SymbolsX, Pilot_SymbolsX, Tx_SymbolsX, Tail_symbolsX]; 
        Tx_SymbolsY_Frame = [Head_SymbolsY, CPC_Pilot_SymbolsY, Pilot_SymbolsY, Tx_SymbolsY, Tail_symbolsY];

        % Used to keep the same channel for the optimization of scaling factor in compensation techniques
        temp_channel_seed = Channel_rand_seed+loop1*1000+4567;
        rng(temp_channel_seed);
%         gpurng(temp_channel_seed);

        Tx_SymbolsX_Frame_PreComp = Tx_SymbolsX_Frame;
        Tx_SymbolsY_Frame_PreComp = Tx_SymbolsY_Frame;

        % DSP up sampling by 2
        Tx_SymbolsX_Dig_upsamples = transpose(upsample(transpose(Tx_SymbolsX_Frame_PreComp),Digital_upsampling_factor)); % Upsampling by 2
        Tx_SymbolsY_Dig_upsamples = transpose(upsample(transpose(Tx_SymbolsY_Frame_PreComp),Digital_upsampling_factor));

        % To keep the power normalized
        Tx_SymbolsX_Dig_upsamples_norm = sqrt(Digital_upsampling_factor).*Tx_SymbolsX_Dig_upsamples;   
        Tx_SymbolsY_Dig_upsamples_norm = sqrt(Digital_upsampling_factor).*Tx_SymbolsY_Dig_upsamples;

        Rx_SymbolsX_Total=[];
        Rx_SymbolsY_Total=[];
for kk=1:1:1         
        % RRC filtering
        Tx_SymbolX_RRC_Filt = conv2(Tx_SymbolsX_Dig_upsamples_norm, rrc_output_tx);
        Tx_SymbolY_RRC_Filt = conv2(Tx_SymbolsY_Dig_upsamples_norm, rrc_output_tx);


        if(Pre_Lby2_CDC)
            Tx_SymbolX_RRC_Filt_CDC = zeros(size(Tx_SymbolX_RRC_Filt));
            Tx_SymbolY_RRC_Filt_CDC = zeros(size(Tx_SymbolY_RRC_Filt));

            for pre_cdc_loop = 1:1:size(Tx_SymbolX_RRC_Filt_CDC, 1)
                [Tx_SymbolX_RRC_Filt_CDC(pre_cdc_loop,:), Tx_SymbolY_RRC_Filt_CDC(pre_cdc_loop,:)] ...
                    = CD_Compensation_Lumped2(Tx_SymbolX_RRC_Filt(pre_cdc_loop,:), Tx_SymbolY_RRC_Filt(pre_cdc_loop,:), Cen_Freqs(pre_cdc_loop),...
                    Digital_upsampling_factor*BaudRate, (No_of_span/2)*SpanLength*1e3, Chrom_Disp, lambda0);
            end 
        else
            Tx_SymbolX_RRC_Filt_CDC = Tx_SymbolX_RRC_Filt;
            Tx_SymbolY_RRC_Filt_CDC = Tx_SymbolY_RRC_Filt;
        end

        % Channel selection and then sum all the sub channels of system 
        Tx_SymbolX_Channel = Channel_sampling_new(Tx_SymbolX_RRC_Filt_CDC, Channel_upsamling_factor, Total_Sub_Channels, BaudRate, WDM_Channel_Spacing, Ch_index, Total_upsamling_factor);
        Tx_SymbolY_Channel = Channel_sampling_new(Tx_SymbolY_RRC_Filt_CDC, Channel_upsamling_factor, Total_Sub_Channels, BaudRate, WDM_Channel_Spacing, Ch_index, Total_upsamling_factor);

        % Fix the transmit power 
        Power_scaling = sqrt((10^((Tx_Pow_dBm-30)/10)*Total_Sub_Channels/2)); % /2 comes due to the divition into 2 polarizations
        % At receiver we receive only one sub channel of a WDM channel   
        Power_scaling_rx = sqrt((10^((Tx_Pow_dBm-30)/10))/2);

        Tx_Symbols_Channel = (Power_scaling).*[Tx_SymbolX_Channel; Tx_SymbolY_Channel]; % scaled accourding to the tx power

        if(Spectrum_plots)
            figure
            my_plot_fft(Tx_Symbols_Channel(1,:), Total_upsamling_factor, BaudRate, 'norm');
        end

         %% Laser Phase Noise         
        if(Laser_Phase_noise == 1)
            Ax = Tx_Symbols_Channel(1,:);
            Ay = Tx_Symbols_Channel(2,:);
            length_Ax = size(Ax);
            length_Ax = length_Ax(2);
            Laser_theta(1, 1) = -pi + (pi+pi)*rand();
            for ii = 1:length_Ax-1
                Laser_w = normrnd(0,sqrt(2*pi*Laser_const/(Total_upsamling_factor*BaudRate)));
                Laser_theta(1, ii+1) = Laser_theta(1,ii) + Laser_w;
            end
            Ax = Ax.*exp(1i*Laser_theta);
            Ay = Ay.*exp(1i*Laser_theta);
            
            Tx_Symbols_Channel = [Ax; Ay];
        end
        
        if(Fiber) % Send via fiber channel SSFM
            Rx_Symbols_Channel = FiberTransmissionLink_SSFM2(Tx_Symbols_Channel,BaudRate, ...
                                    Total_upsamling_factor, lambda0, Chrom_Disp, Gama, ...
                                    loss_dB, Noise_Figure_dB, SpanLength, No_of_span, SSFM_slices_per_span, ...
                                    ASE, PMD_Flag, PMD);
        else
            Rx_Symbols_Channel = Tx_Symbols_Channel;       
        end

        if(Spectrum_plots)
            figure
            my_plot_fft(Rx_Symbols_Channel(1,:), Total_upsamling_factor, BaudRate, 'norm');
        end
%% Laser Phase Noise         
        if(Laser_Phase_noise == 1)
            Ax = Rx_Symbols_Channel(1,:);
            Ay = Rx_Symbols_Channel(2,:);
            length_Ax = size(Ax);
            length_Ax = length_Ax(2);
            Laser_theta(1, 1) = -pi + (pi+pi)*rand();
            for ii = 1:length_Ax-1
                Laser_w = normrnd(0,sqrt(2*pi*Laser_const/(Total_upsamling_factor*BaudRate)));
                Laser_theta(1, ii+1) = Laser_theta(1,ii) + Laser_w;
            end
            Ax = Ax.*exp(1i*Laser_theta);
            Ay = Ay.*exp(1i*Laser_theta);
            
            Rx_Symbols_Channel = [Ax; Ay];
        end

        Rx_SymbolsX_CDC = zeros(No_Sub_Channels, length(Tx_SymbolsX));
        Rx_SymbolsY_CDC = zeros(No_Sub_Channels, length(Tx_SymbolsY));

        sub_ch_loop = 1;%:No_Sub_Channels  % Loop to compute the errors over all sub channel in a WDM system

            % Channel selection (Given WDM channel, loop over all sub channels)
            sub_channel_id = (COI_no-1)*No_Sub_Channels+sub_ch_loop;
            cf = Cen_Freqs(sub_channel_id); 
            tt = (1:size(Rx_Symbols_Channel, 2))/(Total_upsamling_factor*BaudRate);
            Rx_SymbolsX_Channel = Rx_Symbols_Channel(1,:) .* exp(-1i*2*pi*cf*tt);
            Rx_SymbolsY_Channel = Rx_Symbols_Channel(2,:) .* exp(-1i*2*pi*cf*tt);

            % Optical Filtering for center channel
            totalsamples = length(Rx_SymbolsX_Channel);
            f = (-totalsamples/2+1:totalsamples/2)/totalsamples;
            f = fftshift(f) * Total_upsamling_factor*BaudRate; 
            mf = 100;  Bopt = Sub_Channel_Spacing/2; % Bopt = ((1+RRC_beta)*BaudRate)/2;  % 
            Hopt = exp(-(f/Bopt).^(2*mf));
            Rx_SymbolsX_Opt_filt  = ifft(Hopt .* fft(Rx_SymbolsX_Channel));
            Rx_SymbolsY_Opt_filt  = ifft(Hopt .* fft(Rx_SymbolsY_Channel));

            if(Spectrum_plots)
                figure
                my_plot_fft(Rx_SymbolsX_Opt_filt, Total_upsamling_factor, BaudRate, 'norm');
            end

            % Downsampling to 2 sps to do CDC
            Rx_SymbolsX_downsamp_x2=downsample(Rx_SymbolsX_Opt_filt, Channel_upsamling_factor);
            Rx_SymbolsY_downsamp_x2=downsample(Rx_SymbolsY_Opt_filt, Channel_upsamling_factor);

            % CDC
            if(Post_Lby2_CDC)
                [Rx_SymbolsX_CDCt, Rx_SymbolsY_CDCt] = CD_Compensation_Lumped2(Rx_SymbolsX_downsamp_x2,...
                    Rx_SymbolsY_downsamp_x2, cf, Digital_upsampling_factor*BaudRate, ...
                    (No_of_span/2)*SpanLength*1e3, Chrom_Disp, lambda0);      
            elseif(Post_CDC)
                [Rx_SymbolsX_CDCt, Rx_SymbolsY_CDCt] = CD_Compensation_Lumped2(Rx_SymbolsX_downsamp_x2,...
                    Rx_SymbolsY_downsamp_x2, cf, Digital_upsampling_factor*BaudRate, ...
                    No_of_span*SpanLength*1e3, Chrom_Disp, lambda0);
            else
                Rx_SymbolsX_CDCt = Rx_SymbolsX_downsamp_x2;                
                Rx_SymbolsY_CDCt = Rx_SymbolsY_downsamp_x2;            
            end

            % Receiver RRC filter (recover symbols from RRC pulses)
            Rx_SymbolsX_Filt =  conv(Rx_SymbolsX_CDCt, rrc_output_rx); 
            Rx_SymbolsY_Filt =  conv(Rx_SymbolsY_CDCt, rrc_output_rx);

            % Normalized power to keep decoding simpler
            Rx_SymbolsX_Filt_norm = Rx_SymbolsX_Filt./sqrt(Digital_upsampling_factor);
            Rx_SymbolsY_Filt_norm = Rx_SymbolsY_Filt./sqrt(Digital_upsampling_factor);

            % Ignore the head and tail send to mitigate the ssfm residual effects
            Rx_SymbolsX_Filt_short = Rx_SymbolsX_Filt_norm(2*Head_Tail_Len+1:end-2*Head_Tail_Len);
            Rx_SymbolsY_Filt_short = Rx_SymbolsY_Filt_norm(2*Head_Tail_Len+1:end-2*Head_Tail_Len);

            if(Pilot_Len==0)
%                 Rx_SymbolsX = downsample([Rx_SymbolsX_Filt_short((RRC_No_of_taps+rrc2-2)+1:end-(RRC_No_of_taps+rrc2-2))],Digital_upsampling_factor);
%                 Rx_SymbolsY = downsample([Rx_SymbolsY_Filt_short((RRC_No_of_taps+rrc2-2)+1:end-(RRC_No_of_taps+rrc2-2))],Digital_upsampling_factor);
                Rx_SymbolsX_Temp_ds_x1 = downsample(Rx_SymbolsX_Filt_short, Digital_upsampling_factor);
                Rx_SymbolsY_Temp_ds_x1 = downsample(Rx_SymbolsY_Filt_short, Digital_upsampling_factor);

        %% Ignore the filter residual at head and tail
                Rx_SymbolsX_Temp = Rx_SymbolsX_Temp_ds_x1((RRC_No_of_taps+rrc2-2)/2+1:end-(RRC_No_of_taps+rrc2-2)/2);
                Rx_SymbolsY_Temp = Rx_SymbolsY_Temp_ds_x1((RRC_No_of_taps+rrc2-2)/2+1:end-(RRC_No_of_taps+rrc2-2)/2);
                
                Rx_SymbolsX = Rx_SymbolsX_Temp./Power_scaling_rx;
                Rx_SymbolsY = Rx_SymbolsY_Temp./Power_scaling_rx;
            else
                % LMSE used to remove PMD effects(This function also downsamples, nomalize energy and remove the pilots)
                [Rx_SymbolsX, Rx_SymbolsY] = LMS_2x2_PMDEq2(Rx_SymbolsX_Filt_short,...
                Rx_SymbolsY_Filt_short, Pilot_SymbolsX(sub_channel_id, :), Pilot_SymbolsY(sub_channel_id, :),...
                RRC_No_of_taps, Pilot_Len, Power_scaling_rx);
            end
          
            
            % Constallation diagrams after lmse
            if(Constallation_plots)
                scatterplot(Rx_SymbolsX(101:end-100));
                title('X-Pol After LMSE before constant phase correction');
                scatterplot(Rx_SymbolsY(101:end-100));
                title('Y-Pol After LMSE before constant phase correction');
            end

            % Constant Phase Correction
            if(CPC_Pilot_len==0) %Here actual tx signals are used to estimate the phase rotation
                sx= xcorr(Tx_SymbolsX(sub_channel_id, :), Rx_SymbolsX);
                sx_max = max(sx);
                phi_x_cpc = angle (sx_max);
                Rx_SymbolsX_CDC(sub_ch_loop, :) = Rx_SymbolsX;%.*exp(1i*phi_x_cpc);

                sy= xcorr(Tx_SymbolsY(sub_channel_id, :), Rx_SymbolsY);
                sy_max = max(sy);
                phi_y_cpc = angle (sy_max);
                Rx_SymbolsY_CDC(sub_ch_loop, :) = Rx_SymbolsY;%.*exp(1i*phi_y_cpc);
            else % Here the pilots are used for the estmation of phase rotation.
                sx= xcorr(CPC_Pilot_SymbolsX(sub_channel_id, :), Rx_SymbolsX);
                sx_max = max(sx);
                phi_x_cpc = angle (sx_max);
                Rx_SymbolsX_CDC(sub_ch_loop, :) = Rx_SymbolsX.*exp(1i*phi_x_cpc);

                sy= xcorr(CPC_Pilot_SymbolsX(sub_channel_id, :), Rx_SymbolsY);
                sy_max = max(sy);
                phi_y_cpc = angle (sy_max);
                Rx_SymbolsY_CDC(sub_ch_loop, :) = Rx_SymbolsY.*exp(1i*phi_y_cpc);
            end

                   %% BPS
        BPS_Nr_Test_Phase = 40;
        BPS_Moving_Window_Size = 200;
        Rx_SymbolsX_Com = CPR_BPS_v1(Rx_SymbolsX_CDC(sub_ch_loop,:), BPS_Nr_Test_Phase, BPS_Moving_Window_Size, M);
        Rx_SymbolsX_CDC(sub_ch_loop,:) = Rx_SymbolsX_Com.';
        Rx_SymbolsY_Com = CPR_BPS_v1(Rx_SymbolsY_CDC(sub_ch_loop,:), BPS_Nr_Test_Phase, BPS_Moving_Window_Size, M);
        Rx_SymbolsY_CDC(sub_ch_loop,:) = Rx_SymbolsY_Com.';
        size(Rx_SymbolsX_CDC)
        Rx_SymbolsX_Total=[Rx_SymbolsX_Total; Rx_SymbolsX_CDC];
        Rx_SymbolsY_Total=[Rx_SymbolsY_Total; Rx_SymbolsY_CDC];
        
        clear  Tx_SymbolsX_Dig_upsamples_norm_temp;
        clear  Tx_SymbolsY_Dig_upsamples_norm_temp;
        
end
         size(Rx_SymbolsX_Total)
         Rx_SymbolsX_CDC = Rx_SymbolsX_Total;
         Rx_SymbolsY_CDC = Rx_SymbolsY_Total;
        
Rx_SymbolsX_Com = Rx_SymbolsX_CDC(sub_ch_loop, :);        
Rx_SymbolsY_Com = Rx_SymbolsY_CDC(sub_ch_loop, :);        
 
Input_Symbol_real_X = real(Rx_SymbolsX_Com);
Input_Symbol_image_X = imag(Rx_SymbolsX_Com);

Transmit_real_X = real(Tx_SymbolsX(sub_channel_id, :));
Transmit_image_X = imag(Tx_SymbolsX(sub_channel_id, :));

Tx_BitsX_actual=Tx_BitsX(sub_channel_id,:);
Tx_BitsY_actual=Tx_BitsY(sub_channel_id,:);
size(Rx_SymbolsX_Com)
Frame_Len
Rx_SymbolsX_Com_actual=Rx_SymbolsX_Com;
Rx_SymbolsY_Com_actual=Rx_SymbolsY_Com;
Input_Symbol_real_X_actual=Input_Symbol_real_X;
Input_Symbol_image_X_actual=Input_Symbol_image_X;
Transmit_real_X_actual=Transmit_real_X;
Transmit_image_X_actual=Transmit_image_X;


 %% Training data
 Tx_Bits_X=Tx_BitsX_actual(1:log2(M)*Frame_Len);
 Tx_Bits_Y=Tx_BitsY_actual(1:log2(M)*Frame_Len);
 Rx_SymbolsX_Com=Rx_SymbolsX_Com_actual(1:Frame_Len);
 Rx_SymbolsY_Com=Rx_SymbolsY_Com_actual(1:Frame_Len);
 Input_Symbol_real_X=Input_Symbol_real_X_actual(1:Frame_Len);
 Input_Symbol_image_X=Input_Symbol_image_X_actual(1:Frame_Len);
 Transmit_real_X=Transmit_real_X_actual(1:Frame_Len);
 Transmit_image_X=Transmit_image_X_actual(1:Frame_Len);
 fname = "Dataset_16QAM_1000_5WDM32G_PreCDC_test_2_all.mat";
 save(fname,'Tx_Bits_X','Tx_Bits_Y','Rx_SymbolsX_Com','Rx_SymbolsY_Com', 'Tx_Pow_dBm', 'Input_Symbol_real_X','Input_Symbol_image_X','Transmit_real_X','Transmit_image_X');         


    
            % Constallation diagrams after cpc
            if(Constallation_plots)
                scatterplot(Rx_SymbolsX_CDC(100:end-100));
                title('X-Pol After constant phase correction');
                scatterplot(Rx_SymbolsY_CDC(100:end-100));
                title('Y-Pol After constant phase correction');
            end

            % Demodulate CDC symbols and calculate errors
            [errors_cdc(sub_ch_loop, loop1), bit_count_cdc(sub_ch_loop, loop1)] ...
                = Frame_Errors_Cal(Rx_SymbolsX_CDC(sub_ch_loop, :), Rx_SymbolsY_CDC(sub_ch_loop, :), M, Tx_BitsX(sub_channel_id,:),...
                    Tx_BitsY(sub_channel_id,:));

        ber_cdc(:, loop1)= errors_cdc(:, loop1)./bit_count_cdc(:, loop1);
        q_factor_db_cdc(:, loop1) = 20*log10(sqrt(2)*erfcinv(2*ber_cdc(:, loop1)))


    end

