

%% Base system parameters
No_WDM_Channels = 5;            % No of WDM channels
No_Sub_Channels = 1;            % No of sub channels in each WDM channel
Total_Sub_Channels = No_WDM_Channels*No_Sub_Channels;

BaudRate = 32e9;                % Baud rate of each sub channel
RRC_beta = 0.01;                 % Roll-off nyquist pulse of each channel
RRC_No_of_taps =1001;           % Taps of RRC filter
WDM_Channel_Spacing = No_Sub_Channels*BaudRate*1.25;     % Spacing between center of two WDM channels
Sub_Channel_Spacing = BaudRate*1.15;   % Spacing between center of two sub channels

M = 16;                         % QAM modulation

%% Optical link parametrs
Fiber = 1;                      % Enable transmission via fiber
SpanLength = 100.0;              % Span length in km
No_of_span = 10;                % Number of spans
Chrom_Disp = 17.0;              % CD in ps/nm/km
Gama = 1.2;                     % Gama in 1/(W*km)
loss_dB = 0.2;                  % Fiber loss in dB/km
Noise_Figure_dB = 6;          % EDFA noisefigure in dB
lambda0  = 1552.93;                % wavelenght of center WDM channel in nm
PMD = 0.1;                      % in ps/km^(1/2)
ASE = 1;                        % Turn on(1) and off(0 or else) ASE noise effects
PMD_Flag = 1;                   % Turn on(1) and off(0 or else) PMD effects
SSFM_slices_per_span = 200;     % Number of steps used in SSFM for each span
Laser_Phase_noise = 1;
Laser_const = 100000;

%% Simulation variables 
% Frame structure [Head, Phase_comp_pilot, LMSE_pilot, Data_Frame, Tail]
Frame_Len = 2^15*10;               % Each loop carries one frame
Pilot_Len = 2^14;               % Pilot length for LMSE
CPC_Pilot_len = 0;        % Pilot length for constant phase correction
Head_Tail_Len = 2^14;            % Head and tail used to ignore the ssfm residual effects

NoLoops = 1;                    % Number of frame simulated for each tx power value

Data_rand_seed = 221185960;     % Seed used to control the randamness of data
Channel_rand_seed = 695811220;  % Seed used to control the randomness of channel randomness

%% Flags
Spectrum_plots = 0;             % To see the spectrum at different stages of simulation
Constallation_plots = 0;        % To see the constallation at different points of simulation
Save_Flag = 0;                  % Save the Q-factor results

Pre_Lby2_CDC = 1;
Post_Lby2_CDC = 1
Post_CDC = 0;

CDC_ONLY_Flag = 1;              




%% Basic calculations
Digital_upsampling_factor = 2;
Total_upsamling_factor = 4*2^ceil(log2(((No_WDM_Channels-1)*WDM_Channel_Spacing+(No_Sub_Channels-1)*Sub_Channel_Spacing+(1+RRC_beta)*BaudRate)/BaudRate));
Channel_upsamling_factor = Total_upsamling_factor/Digital_upsampling_factor;

%wdm_ch_ind = floor((0:No_WDM_Channels-1)-(No_WDM_Channels-1)/2);
%sub_ch_ind = (Sub_Channel_Spacing/WDM_Channel_Spacing).*floor((0:No_Sub_Channels-1)-(No_Sub_Channels-1)/2);
wdm_ch_index = (0:No_WDM_Channels-1)-(No_WDM_Channels-1)/2;
sub_ch_index = (Sub_Channel_Spacing/WDM_Channel_Spacing).*((0:No_Sub_Channels-1)-(No_Sub_Channels-1)/2);

Ch_index = kron(ones(size(wdm_ch_index)), sub_ch_index)+kron(wdm_ch_index, ones(size(sub_ch_index)));
Cen_Freqs = Ch_index.*WDM_Channel_Spacing;        

COI_no = floor((No_WDM_Channels+1)/2) % set base band channel as WDM channel as COI
%SCOI_no = floor((Total_Sub_Channels+1)/2); % set a sub channel index or set


% RRC Channel filter taps
rrc_output_tx = RC_RRC_for_FTN(RRC_beta, RRC_No_of_taps,'sqrt',1/Digital_upsampling_factor);

% RRC receive filter
rrc2 = RRC_No_of_taps;
RRC_beta2 = RRC_beta;
rrc_output_rx=RC_RRC_for_FTN(RRC_beta, RRC_No_of_taps,'sqrt',1/Digital_upsampling_factor);



%{
Disp_parameters = 1;
if(Disp_parameters)
disp('<strong>System Parameters</strong>');
disp(['No of channels   :', num2str(No_sub_channels)]);
disp(['Baud rate        :', num2str(BaudRate/1e9), ' GBd']);
disp(['Channel spacing  :',num2str(Channel_Spacing/1e9), ' GHz']);
disp(['Modulation       :', num2str(M), ' QAM' ]);


disp('<strong>Simulation Parameters</strong>');
disp(['No of loops      :', num2str(NoLoops)]);
disp('<strong>Simulation Frame Structure</strong>');
disp(['Head             :', num2str(Head_Tail_Len)]);
disp(['Pilots LMSE      :', num2str(Pilot_Len)]);
disp(['Pilots Phase Est :', num2str(Phase_Com_Pilot_len)]);
disp(['Symbols          :', num2str(Frame_Len)]);
disp(['Tail             :', num2str(Head_Tail_Len)]);

disp('<strong>Fiber Parameters</strong>');
disp(['Fiber length         :', num2str(No_of_span), 'x', num2str(SpanLength), ' km']);
disp(['CD coeff             :', num2str(Chrom_Disp), ' ps/(nm*km)']);
disp(['Center wave length   :', num2str(lambda0), ' nm']);
disp(['Gamma                :', num2str(Gama), ' 1/(W*km)']);
disp(['EDFA NF              :', num2str(Noise_Figure_dB), ' dB Enabled=', num2str(AWGN)])
disp(['PMD coeff            :', num2str(PMD), ' ps/km^(1/2) Enabled=', num2str(PMD_Flag)])
disp(['Attunation           :', num2str(loss_dB), ' dB/km']);
disp(['SSFM steps per span  :', num2str(SSFM_slices_per_span)]);
disp(['Total sampling       :', num2str(Total_upsamling_factor), ' (Digital sampling ', num2str(Digital_upsampling_factor), ' x Channel over sampling ', num2str(Channel_upsamling_factor), ')'])


disp('<strong>Filter Parameters</strong>');
disp(['RRC Roll-off    :', num2str(RRC_beta), '     RRC filter taps     :', num2str(RRC_No_of_taps),]);
end
%}
