clc
clear all
close all

No_sub_channels = 1; % Only one channel is considered now
No_of_code_blocks = 1; % The block consider for a single SSFM
Tx_Pow_dBm = [1] %Transmission power per channel
BaudRate = 32e9;
Channel_Spacing = 50e9;
RRC_beta = 0.01;
RRC_No_of_taps =1001;
RRC_tau = 1.0;
FP_xi = Channel_Spacing/(BaudRate*(1+RRC_beta));
Total_upsamling_factor = 4*2^ceil(log2((1+RRC_beta)*((No_sub_channels-1)*FP_xi+1)));
Digital_upsampling_factor = 2;
Channel_upsamling_factor = Total_upsamling_factor/Digital_upsampling_factor;

%% Dataset size (This represents the symbol size in each block of data generated)

N_data=2^15;
%% Pilot and head and tail
Pilot_Len = 0;%10000;
Phase_Com_Pilot_len = 0;%100;
Head_Tail_Len = 0;%100;

%% Optical link parametrs
Chrom_Disp = 17.0;                      % CD in ps/nm/km
Gama = 1.2;                         % Gama in 1/(W*km)
AWGN = 1;                               % Turn on(1) and off(0 or else) AWGN
PMD_Flag = 0;
PMD = 0.1;
loss_dB = 0.2;                          % Fiber loss in dB/km
lambda0  = 1552.93;                     % wavelenght in nm
Noise_Figure_dB = 6;              % EDFA noisefigure in dB
SpanLength = 80.0;                 % Span length in km
No_of_span = 15.0;  
Transmission_length=SpanLength*No_of_span % Number of spans
perturb_symbol_legth=100;            % Perturbation symbol length
epslon=1.2000;

%% Modulation Demodulation Initialization
M=64;

%% Interleaver Initilization
% Random interleaver
Intlv_State = randi([0, 2^15], No_sub_channels, 2); %randi([0,2^32]);  % Interleaver can take any state betweeen 0, 2^32

%% LDPC Code Encoding Decding Initialization
Pr = log2(M)*N_data; % The number six represents the total number of blocks of data each with a size 2^15 
                       %(5 of which is used for the training and 1 is for the testing)

%% RRC Channel filter initiation
rrc_output_tx = RC_RRC_for_FTN(RRC_beta, RRC_No_of_taps, RRC_tau,'sqrt',1/Digital_upsampling_factor);


%% RRC receive filter
rrc2 = RRC_No_of_taps;
RRC_beta2 = RRC_beta;
rrc_output_rx=RC_RRC_for_FTN(RRC_beta2, rrc2, RRC_tau,'sqrt',1/Digital_upsampling_factor);

BER = zeros(1, length(Tx_Pow_dBm));
Tx_SNR_dB = zeros(1, length(Tx_Pow_dBm));
Rx_SNR_dB = zeros(1, length(Tx_Pow_dBm));

for pow_loop = 1:1:length(Tx_Pow_dBm) %length(Tx_Pow_dBm)

    Total_Errors = 0;
    Total_Bits = 0;
    bool =  true;
    loop_count = 0;
%     rng('default')
% ss=rng  
        
       %% Initilizations and QAM symbol frame generation
       
       % IMPORTANT: In the generation of the training and testing data, the
       % information bits are saved first in the folder
       % "C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\MATLAB_CODE".
       % Please change this address in your case. This is done to
       % facilitate the transmission of the SAME training data five times
       % and taking the average for denoising. 
       
        Tx_BitsX = zeros(No_sub_channels, Pr*No_of_code_blocks); 
        Tx_BitsY = zeros(No_sub_channels, Pr*No_of_code_blocks);
        Tx_SymbolsX = zeros(No_sub_channels, Pr*No_of_code_blocks/log2(M));
        Tx_SymbolsY = zeros(No_sub_channels, Pr*No_of_code_blocks/log2(M));
        
%% ---------------------------------Please activate this section if you want to generate new information bits----------------------- %% 

%         for sub_ch_loop = 1: No_sub_channels
%             Tx_BitsX(sub_ch_loop, :) = randi([0,1], 1, Pr*No_of_code_blocks); % Random bit generation
%             Tx_BitsY(sub_ch_loop, :) = randi([0,1], 1, Pr*No_of_code_blocks); % Random bit generation
%             
% 
%             Tx_SymbolsX(sub_ch_loop, :) = transpose(qammod(Tx_BitsX(sub_ch_loop, :).', M,  'InputType', 'bit', 'UnitAveragePower', true));  %M QAM modulation 
%             Tx_SymbolsY(sub_ch_loop, :) = transpose(qammod(Tx_BitsY(sub_ch_loop, :).', M,  'InputType', 'bit', 'UnitAveragePower', true));   %M QAM modulation       
%         end
%         save ('Tx_BitsX.txt', 'Tx_BitsX','-ascii');
%         save ('Tx_BitsY.txt', 'Tx_BitsY','-ascii');

%% ---------------------------------------------------------------------------------------------------------------------------------- %% 

        load ('Tx_BitsX.txt', 'Tx_BitsX','-ascii');
        load ('Tx_BitsY.txt', 'Tx_BitsY','-ascii');
%        (log2(M)*5*N_data+1:end)
Tx_BitsX = Tx_BitsX((log2(M)*3*N_data+1:end-(log2(M)*2*N_data)));
Tx_BitsY = Tx_BitsY((log2(M)*3*N_data+1:end-(log2(M)*2*N_data)));
        Tx_SymbolsX(1, :) = transpose(qammod(Tx_BitsX.', M,  'InputType', 'bit', 'UnitAveragePower', true));  %M QAM modulation 
        Tx_SymbolsY(1, :) = transpose(qammod(Tx_BitsY.', M,  'InputType', 'bit', 'UnitAveragePower', true));   %M QAM modulation       
        
%% ---------------------------------Please activate this section if you want to generate new training bits----------------------- %% 

        
%         if(Pilot_Len>0)
%             Pilot_SymbolsX_bit=randi([0,M-1], No_sub_channels, Pilot_Len);
%             Pilot_SymbolsY_bit=randi([0,M-1], No_sub_channels, Pilot_Len);
% %             Pilot_SymbolsX = qammod(randi([0,M-1], No_sub_channels, Pilot_Len), M,  'InputType', 'integer', 'UnitAveragePower', true);
% %             Pilot_SymbolsY = qammod(randi([0,M-1], No_sub_channels, Pilot_Len), M,  'InputType', 'integer', 'UnitAveragePower', true);
% %                       
%         else

        Pilot_SymbolsX = []; %qammod(Pilot_SymbolsX_bit, M,  'InputType', 'integer', 'UnitAveragePower', true);
        Pilot_SymbolsY = []; %qammod(Pilot_SymbolsY_bit, M,  'InputType', 'integer', 'UnitAveragePower', true);

            
        if(Head_Tail_Len>0)
            Head_SymbolsX = qammod(randi([0,M-1], No_sub_channels, Head_Tail_Len), M,  'InputType', 'integer', 'UnitAveragePower', true);
            Tail_symbolsX = qammod(randi([0,M-1], No_sub_channels, Head_Tail_Len), M,  'InputType', 'integer', 'UnitAveragePower', true);
            Head_SymbolsY = qammod(randi([0,M-1], No_sub_channels, Head_Tail_Len), M,  'InputType', 'integer', 'UnitAveragePower', true);
            Tail_symbolsY = qammod(randi([0,M-1], No_sub_channels, Head_Tail_Len), M,  'InputType', 'integer', 'UnitAveragePower', true);
        else
            Head_SymbolsX = [];
            Tail_symbolsX = [];
            Head_SymbolsY = [];
            Tail_symbolsY = [];       
        end
        
        Tx_SymbolsX_Frame = [Head_SymbolsX, Pilot_SymbolsX, Tx_SymbolsX, Tail_symbolsX]; 
        Tx_SymbolsY_Frame = [Head_SymbolsY, Pilot_SymbolsY, Tx_SymbolsY, Tail_symbolsY];

        
        COI_no =(No_sub_channels+1)/2;  % Selection of the channel of interest
        
      
        %% DSP up sampling by 2
        Tx_SymbolsX_Dig_upsamples = transpose(upsample(transpose(Tx_SymbolsX_Frame),Digital_upsampling_factor)); % Upsampling by 2
        Tx_SymbolsY_Dig_upsamples = transpose(upsample(transpose(Tx_SymbolsY_Frame),Digital_upsampling_factor));
        
        %% To keep the power normalized
        Tx_SymbolsX_Dig_upsamples_norm = sqrt(Digital_upsampling_factor).*Tx_SymbolsX_Dig_upsamples;   
        Tx_SymbolsY_Dig_upsamples_norm = sqrt(Digital_upsampling_factor).*Tx_SymbolsY_Dig_upsamples;
        
        Rx_SymbolsX_Filt_norm=[];
        Rx_SymbolsY_Filt_norm=[];
        
for kk=1:1:1
             
        Tx_SymbolsX_Dig_upsamples_norm_temp=Tx_SymbolsX_Dig_upsamples_norm;
        Tx_SymbolsY_Dig_upsamples_norm_temp=Tx_SymbolsY_Dig_upsamples_norm;
             
        %% RRC filtering
        Tx_SymbolX_RRC_Filt = conv2(Tx_SymbolsX_Dig_upsamples_norm_temp, rrc_output_tx); % rrc pulse filtering
        Tx_SymbolY_RRC_Filt = conv2(Tx_SymbolsY_Dig_upsamples_norm_temp, rrc_output_tx);

%% CD compensation
        CDC_flag=(Chrom_Disp~=0);
        [Tx_SymbolX_RRC_Filt, Tx_SymbolY_RRC_Filt] = CD_Compensation_Lumped(CDC_flag, Tx_SymbolX_RRC_Filt, Tx_SymbolY_RRC_Filt, 0, Digital_upsampling_factor*BaudRate, No_of_span*SpanLength*10^3, 0.5*Chrom_Disp, lambda0);
        
        %% Channel select and sum the signals
        Tx_SymbolX_Channel = Channel_sampling(Tx_SymbolX_RRC_Filt, Channel_upsamling_factor, No_sub_channels, RRC_beta, FP_xi, RRC_tau, Total_upsamling_factor, RRC_No_of_taps); % Upsample, frequency shift and combine
        Tx_SymbolY_Channel = Channel_sampling(Tx_SymbolY_RRC_Filt, Channel_upsamling_factor, No_sub_channels, RRC_beta, FP_xi, RRC_tau, Total_upsamling_factor, RRC_No_of_taps);
        
        %% Fix the transmit power
        Power_scaling = sqrt((10^((Tx_Pow_dBm(pow_loop)-30)/10)*(No_sub_channels)/2)); % /2 comes due to the divition into 2 polarizations
        Power_scaling_rx = sqrt((10^((Tx_Pow_dBm(pow_loop)-30)/10)/2));
        
        Tx_Symbols_Channel = (Power_scaling).*[Tx_SymbolX_Channel; Tx_SymbolY_Channel]; % scaled accourding to the tx power
        
        figure
        my_plot_fft(Tx_Symbols_Channel(1,:), Total_upsamling_factor, BaudRate, 'norm')
        
        
          Rx_Symbols_Channel_x=[];
          Rx_Symbols_Channel_y=[];
    
   
        %% Send via fiber channel SSFM        
        %{%
        Rx_Symbols_Channel = FiberTransmissionLink_SSFM(Tx_Symbols_Channel,BaudRate, ...
                                Total_upsamling_factor, lambda0, Chrom_Disp, Gama, ...
                                loss_dB, Noise_Figure_dB, SpanLength, No_of_span, AWGN, PMD_Flag, PMD);  
                                      
        figure
        my_plot_fft(Rx_Symbols_Channel(1,:), Total_upsamling_factor, BaudRate, 'norm')

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
        figure
        my_plot_fft(Rx_SymbolsX_Opt_filt, Total_upsamling_factor, BaudRate, 'norm')

        %% Down sample to x2 sample domain
        Rx_SymbolsX_downsamp_x2=downsample(Rx_SymbolsX_Opt_filt, Channel_upsamling_factor); % Downsampling
        Rx_SymbolsY_downsamp_x2=downsample(Rx_SymbolsY_Opt_filt, Channel_upsamling_factor);
        
        Rx_SymbolsXY = [Rx_SymbolsX_downsamp_x2; Rx_SymbolsY_downsamp_x2];
        BP_Nslices = 1;
        %% CD compensation
        CDC_flag=(Chrom_Disp~=0);
        [Rx_SymbolsX_CDC, Rx_SymbolsY_CDC] = CD_Compensation_Lumped(CDC_flag, Rx_SymbolsX_downsamp_x2, Rx_SymbolsY_downsamp_x2, cf, Digital_upsampling_factor*BaudRate, No_of_span*SpanLength*10^3, 0.5*Chrom_Disp, lambda0);
%        Phi_opt(pow_loop) = DBP_Optimization(Rx_SymbolsXY, BaudRate, Digital_upsampling_factor,...
%                                 BP_Nslices, lambda0, Chrom_Disp, Gama, loss_dB, SpanLength,...
%                                 No_of_span, rrc_output_rx, RRC_No_of_taps, rrc2, Power_scaling_rx, ...
%                                 Tx_BitsX, Tx_BitsY, Tx_SymbolsX, Tx_SymbolsY);
        %% Receiver RRC filter (invers tx RRC filter)
        Rx_SymbolsX_Filt =  conv(Rx_SymbolsX_CDC, rrc_output_rx);   % Receiver filtering for baseband center channel
        Rx_SymbolsY_Filt =  conv(Rx_SymbolsY_CDC, rrc_output_rx);

        Rx_SymbolsX_Filt_norm_temp = Rx_SymbolsX_Filt./sqrt(Digital_upsampling_factor);
        Rx_SymbolsY_Filt_norm_temp = Rx_SymbolsY_Filt./sqrt(Digital_upsampling_factor);

        Rx_SymbolsX_Filt_norm=[Rx_SymbolsX_Filt_norm; Rx_SymbolsX_Filt_norm_temp];
        Rx_SymbolsY_Filt_norm=[Rx_SymbolsY_Filt_norm; Rx_SymbolsY_Filt_norm_temp];
        
        clear  Tx_SymbolsX_Dig_upsamples_norm_temp;
        clear  Tx_SymbolsY_Dig_upsamples_norm_temp;
        
end
       
        %% Without LMSE
        %{%
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

%         %% BPS
%         BPS_Nr_Test_Phase = 512;
%         BPS_Moving_Window_Size = 60;
%         Rx_SymbolsX_Com = CPR_BPS_v1(Rx_SymbolsX_Com(1,:), BPS_Nr_Test_Phase, BPS_Moving_Window_Size, M);
%         Rx_SymbolsX_Com = Rx_SymbolsX_Com.';
%         Rx_SymbolsY_Com = CPR_BPS_v1(Rx_SymbolsY_Com(1,:), BPS_Nr_Test_Phase, BPS_Moving_Window_Size, M);
%         Rx_SymbolsY_Com = Rx_SymbolsY_Com.';
%         
%         %% Plots Before LMSE
%         %Constallation diagrams
%     
%         scatterplot(Rx_SymbolsX_Com);
%         title('X-Pol Before LMSE')
%         scatterplot(Rx_SymbolsY_Com);
%         title('Y-Pol Before LMSE')
    
%         %% LMS Equalization
%         
%         Rx_SymbolsX_Filt_short = Rx_SymbolsX_Filt_norm(2*Head_Tail_Len+1:end-2*Head_Tail_Len);
%         Rx_SymbolsY_Filt_short = Rx_SymbolsY_Filt_norm(2*Head_Tail_Len+1:end-2*Head_Tail_Len);
%             
%         [Rx_SymbolsX,Rx_SymbolsY,MSEX,MSEY] = LMS_2x2_PMDEq2(Rx_SymbolsX_Filt_short,Rx_SymbolsY_Filt_short,...
%                                     Pilot_SymbolsX(COI_no, :), Pilot_SymbolsY(COI_no, :), RRC_No_of_taps, Pilot_Len, Power_scaling_rx);
% 
%         Rx_SymbolsX_Scaled = Rx_SymbolsX./Power_scaling_rx; % Power scaling to make it unit
%         Rx_SymbolsY_Scaled = Rx_SymbolsY./Power_scaling_rx;
%         
%         Rx_SymbolsX_Com = Rx_SymbolsX_Scaled;
%         Rx_SymbolsY_Com = Rx_SymbolsY_Scaled;
%         
%         %% Plots After LMSE
%         %Constallation diagrams
%        
%         scatterplot(Rx_SymbolsX_Com);
%         title('X-Pol After LMSE')
%         scatterplot(Rx_SymbolsY_Com);
%         title('Y-Pol After LMSE')
        
        
%% Selection of the received symbols and the corresponding transmitted symbols to generate the dataset

Input_Symbol_real_X = real(Rx_SymbolsX_Com);
Input_Symbol_image_X = imag(Rx_SymbolsX_Com);

Transmit_real_X = real(Tx_SymbolsX);
Transmit_image_X = imag(Tx_SymbolsX);


% %% Dataset for DNN testing
% 
% Tx_BitsX_actual=Tx_BitsX;
% Tx_BitsY_actual=Tx_BitsY;
% Rx_SymbolsX_Com_actual=Rx_SymbolsX_Com;
% Rx_SymbolsY_Com_actual=Rx_SymbolsY_Com;
% Input_Symbol_real_X_actual=Input_Symbol_real_X;
% Input_Symbol_image_X_actual=Input_Symbol_image_X;
% Transmit_real_X_actual=Transmit_real_X;
% Transmit_image_X_actual=Transmit_image_X;
        
 
Input_Symbol_real_X = real(Rx_SymbolsX_Com);
Input_Symbol_image_X = imag(Rx_SymbolsX_Com);

Transmit_real_X = real(Tx_SymbolsX);
Transmit_image_X = imag(Tx_SymbolsX);

        
%% Dataset for DNN training

% IMPORTANT: The training data is saved in a folder with path
% "C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\DATASET\TRAINING_DATA".
% Please change this if you want to save in a different folder.

Tx_BitsX_actual=Tx_BitsX;
Tx_BitsY_actual=Tx_BitsY;
Rx_SymbolsX_Com_actual=Rx_SymbolsX_Com;
Rx_SymbolsY_Com_actual=Rx_SymbolsY_Com;
Input_Symbol_real_X_actual=Input_Symbol_real_X;
Input_Symbol_image_X_actual=Input_Symbol_image_X;
Transmit_real_X_actual=Transmit_real_X;
Transmit_image_X_actual=Transmit_image_X;


% Training data
% Tx_BitsX=Tx_BitsX_actual((log2(M)*5*N_data+1:end));
% Tx_BitsY=Tx_BitsY_actual((log2(M)*5*N_data+1:end));
Rx_SymbolsX_Com=Rx_SymbolsX_Com_actual;
Rx_SymbolsY_Com=Rx_SymbolsY_Com_actual;
Input_Symbol_real_X=Input_Symbol_real_X_actual;
Input_Symbol_image_X=Input_Symbol_image_X_actual;
Transmit_real_X=Transmit_real_X_actual;
Transmit_image_X=Transmit_image_X_actual;

save('D:\MASC\Perturbation_based\learned_AM_Model\Dataset\Dataset_Dual_train_1_PB','Tx_BitsX','Tx_BitsY','Rx_SymbolsX_Com','Rx_SymbolsY_Com', 'Tx_Pow_dBm', 'Input_Symbol_real_X','Input_Symbol_image_X','Transmit_real_X','Transmit_image_X');         
   
end


