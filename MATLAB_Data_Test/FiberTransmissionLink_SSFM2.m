function f = FiberTransmissionLink_SSFM2(Axy,BaudRate, Total_upsamling_factor,...
    lambda0, Chrom_Disp, Gama, loss_dB, Noise_Figure, SpanLength, No_of_span, nbr_of_slices,...
    ASE, PMD_Flag, PMD)
%rng(563259445);
Ax_ = Axy(1,:); Ay_ = Axy(2,:);
% Ax_ = gpuArray(Ax); Ay_ = gpuArray(Ay);

% Constants/Constantes

%i=sqrt(-1);
c=3e8;
h = 6.626068e-34;                       % Planck constant

beta2 = -(lambda0*1e-9)^2/(2*pi*c) * Chrom_Disp*1e-6;         % s^2/m
nu0 = c/(lambda0*1e-9);

totalsamples = length(Ax_);
f = (-totalsamples/2+1:totalsamples/2)/totalsamples;
f = fftshift(f)*Total_upsamling_factor*BaudRate; 
w = 2*pi.*f; 


loss = loss_dB * log(10)/10 * 1e-3; 

% EDFA parameters
Gain1_dB = loss_dB*SpanLength;  % Noise figure in dB
gain1 = 10^(Gain1_dB/10);  
NF1 = 10^(Noise_Figure/10);

%Pase1 = (gain1-1)*h*nu0*(NF1/2)*Total_upsamling_factor*BaudRate; % Pase is the Power of ASE noise 
Pase1 = (gain1-1)*h*nu0*(NF1/2)*Total_upsamling_factor*BaudRate;% The Power of ASE noise per polarization \ per span 


%nbr_of_slices = 1000;
dL = SpanLength*1e3/nbr_of_slices;
if(loss_dB~=0)
    L_eff=((1-exp(-loss*dL))/loss);
else
    L_eff = dL;
end

% dispersion
Disp_ = exp(-1i/2 * beta2 .* w.^2 .* dL); % '+' or '-'?: '-' is for positive Dispersion.
% Disp_ = gpuArray(Disp);


if No_of_span >= 1

    for iter_span = 1:No_of_span

        for iter = 1:nbr_of_slices
                        
            AX = fft(Ax_); AY = fft(Ay_);
            
            % dispersion
            AX = AX .* Disp_; AY = AY .* Disp_;
                        
            if(PMD_Flag ==1)
                theta = (2*pi)*rand(1);
                %theta = 0.0;                          % No polarisation mixing    
                AX_inter = cos(theta).*AX + sin(theta).*AY;
                AY_inter = -sin(theta).*AX + cos(theta).*AY;
            
                % Phase shift between the two axes of the sections
                % /Dephasage entre les 2 axes propres du troncons
                phi = 2*pi*rand(1);
                %phi = 0.0; %For no mixing between XY    
                AX_inter = AX_inter .* exp(1i*phi/2);
                AY_inter = AY_inter .* exp(-1i*phi/2);
            
                % Birefringence application/Application de la birefringence
                %tau = PMD*1e-12 * 1e-3*dL; % Deterministic PMD (in ps/km)/PMD deterministe (en ps/km)
                tau = sqrt((PMD*1e-12)^2 * 1e-3*dL) * randn(1); % PMD random (in ps/sqrt(km))/PMD aleatoire (en ps/sqrt(km))
                AX_inter = exp(1i .*w .* tau/2) .* AX_inter;
                AY_inter = exp(-1i .*w .* tau/2) .* AY_inter;
            
                % Reverse rotation of the polarization for return in the axes of laboratory/Rotation inverse de la polarisation pour retour dans les axes du laboratoire
                AX = cos(-theta).*AX_inter + sin(-theta).*AY_inter;
                AY = -sin(-theta).*AX_inter + cos(-theta).*AY_inter;
            end
            
            Ax_ = ifft(AX); Ay_ = ifft(AY);

            % attenuation & NL
            Ax_ = Ax_.* exp(-loss/2 *dL);
            Ay_ = Ay_.* exp(-loss/2 *dL);
            
            % nonlinear effects
            Ax_inter = Ax_; Ay_inter = Ay_;

            Ax_ = Ax_inter.* exp(1i.*(8/9).*Gama*1e-3 .* (abs(Ax_inter).^2 + abs(Ay_inter).^2) .* L_eff);
            Ay_ = Ay_inter.* exp(1i.*(8/9).*Gama*1e-3 .* (abs(Ay_inter).^2 + abs(Ax_inter).^2) .* L_eff);
            
        end

        % 1st stage EDFA

        % **************
               
        
        Ax_ = sqrt(gain1).*Ax_; Ay_ = sqrt(gain1).*Ay_;

        % noise/bruit ASE 1st EDFA
        if(ASE == 1)
            asenoise_x = sqrt(Pase1/2).*(randn(1,totalsamples)+1i.*randn(1,totalsamples));
            asenoise_y = sqrt(Pase1/2).*(randn(1,totalsamples)+1i.*randn(1,totalsamples));
            %load('ase_noise.mat');

            Ax_ = Ax_ + asenoise_x;
            Ay_ = Ay_ + asenoise_y;
            
        end


    end   

end


% Ax = gather(Ax_); Ay = gather(Ay_);
f = [Ax_; Ay_];
end

