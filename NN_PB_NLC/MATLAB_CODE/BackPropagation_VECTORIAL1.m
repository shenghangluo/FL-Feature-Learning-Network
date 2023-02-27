function BPoutput_xy = BackPropagation_VECTORIAL1(BPinput_xy,BaudRate,Digital_upsampling_factor,...
                                Phi, BP_Nslices, lambda0, Chrom_Disp, Gama, ...
                                            loss_dB, SpanLength, No_of_span)

Ax = BPinput_xy(1,:); Ay = BPinput_xy(2,:);

% Constants/Constantes
i=sqrt(-1);
c=3e8;
%h = 6.626068e-34;                       % Planck constant

                
% disp(sprintf('Wavelength = %3g nm',lambda0));

% definition of the frequency axis/definition de l'axe des frequences
totalsamples = length(Ax);
f = (-totalsamples/2+1:totalsamples/2)/totalsamples;
f = fftshift(f) * BaudRate*Digital_upsampling_factor; 
w = 2*pi.*f; 


% Fibre transmission parameters
% *****************************
                          % Chromatic dispersion in s/m/m
beta2 = -(lambda0*1e-9)^2/(2*pi*c) * Chrom_Disp*1e-6;         % s^2/m

loss = loss_dB * log(10)/10 * 1e-3;             % 1/m

beta2BP = -beta2;
GamBP = -Gama*1e-3;

% EDFA parameters
% ***************
Gain1_dB = loss_dB*SpanLength;                   % Noise figure in dB
gain1 = 10^(Gain1_dB/10);  


% Input power adjustment
% **********************
% BP is similar to optical phase conjugaison
%   -> the power increased along the "fiber" line
Ax = Ax/sqrt(gain1);
Ay = Ay/sqrt(gain1);



dL = SpanLength*1e3/BP_Nslices;
%dLeff = (1 - exp(-loss*dL))/loss;
if(loss_dB~=0)
    dLeff=(1-exp(-loss*dL))/loss;
else
    dLeff=dL;
end
%disp(' ' )
disp(sprintf('BP span number (%2dspans x %3dKm)', No_of_span, SpanLength))
if No_of_span >= 1
   
    for iter_span = 1:No_of_span
       %wb = waitbar(0,'Please wait'); 
       % disp(sprintf('   %2dth BP section',iter_span))
        
        % transmission in Fiber span
        % **************************
        
        % propagation in transmission fiber
        for iter = 1:BP_Nslices
            %waitbar(iter / BP_Nslices)
           
            AX = fft(Ax); AY = fft(Ay);%% A\delta/2
            
            % dispersion
            Disp = exp(-i/2 * beta2BP .* w.^2 .* dL); %% Diagonal 
            AX = AX .* Disp; AY = AY .* Disp;  
            
            Ax_inter = ifft(AX); Ay_inter = ifft(AY);%% A\delta/2
                                    
            % attenuation & NL
            RelativePower = 10^((iter/BP_Nslices)*Gain1_dB/10);       % Increased power along the "fiber" 

                  %% RelativePower = 10^((1-(iter/BP_Nslices))*Gain1_dB/10);   % Decreased power along the "fiber" 
            Ax = Ax_inter .* exp(i*Phi*GamBP*RelativePower .* (abs(Ax_inter).^2 + abs(Ay_inter).^2) .* dLeff);
            Ay = Ay_inter .* exp(i*Phi*GamBP*RelativePower .* (abs(Ay_inter).^2 + abs(Ax_inter).^2) .* dLeff);
%             Ax = Ax_inter .* exp(i*Phi*GamBP*RelativePower .* (abs(Ax_inter).^2) .* dLeff);
%             Ay = Ay_inter .* exp(i*Phi*GamBP*RelativePower .* (abs(Ay_inter).^2) .* dLeff);

%             Ax = Ax_inter;
%             Ay = Ay_inter;
        end;
        %close(wb);
               
    end;
    
end;
Ax = Ax.*sqrt(gain1);
Ay = Ay.*sqrt(gain1);
BPoutput_xy = [Ax; Ay];
