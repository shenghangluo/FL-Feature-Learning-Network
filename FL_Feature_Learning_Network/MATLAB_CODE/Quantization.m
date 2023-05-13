clear all
close all
clc

weight_1=csvread('weight_8.csv');
bias_1=csvread('weight_9.csv');
weight_2=csvread('weight_10.csv');
bias_2=csvread('weight_11.csv');
weight_3=csvread('weight_12.csv');
bias_3=csvread('weight_13.csv');

load("Processed_state.mat")

% Quantization Level
l=75;

[idx,C] = kmeans(weight_1(:,1),l);
[idx2,C2] = kmeans(weight_1(:,2),l);
x=[C(idx), C2(idx2)];

out_1 = x.'*state.'+bias_1;
out_2 = zeros(size(out_1));
out_2(out_1>=0) = out_1(out_1>=0);
out_2(out_1<0)  = 0.5*out_1(out_1<0);

out_2 = weight_2.'*out_2+bias_2;
out_3 = zeros(size(out_2));
out_3(out_2>=0) = out_2(out_2>=0);
out_3(out_2<0)  = 0.5*out_2(out_2<0);

out_4 = weight_3.'*out_3+bias_3;

out_real = out_4(1,:);
out_image = out_4(2,:);

save('Processed_out.mat', 'out_real', 'out_image')


clear all
close all
clc 
load('Prossed_X_Dataset_16QAM_1000_PreCDC_5WDM32G_test_2.mat')
N_symb=65536;

load('Processed_out.mat')
data_real=out_real;
data_image=out_image;

data_real = reshape(data_real.', [1,N_symb]);
data_image = reshape(data_image.', [1,N_symb]);
COI_no=1;
M=16;
Total_Errors = 0;
loop_count = 0;
Total_Bits = 0;
pow_loop = 1;
epslon_value=[1];
 BER_min=[];
 Q_max=[];
for i=1:1:length(epslon_value)
    epslon=epslon_value(i);

%%-----------------------------------------------------------------------------------
data_complex = complex(data_real, data_image);
data_complex = reshape(data_complex.', [1,N_symb]);

alpha=10^(0.1*(2-(2)));

Rx_SymbolsX_Com1 = Rx_SymbolsX_Com-data_complex;
Tx_SymbolsX = complex(Transmit_real_X, Transmit_image_X);

Trans = complex(Transmit_real_X, Transmit_image_X);
Loss = mean(abs((Rx_SymbolsX_Com1(1:end) - Trans(1:end))).^2);

scatterplot(Rx_SymbolsX_Com1)
        %% QAM Demapper
        Rx_BitsX=transpose(qamdemod(Rx_SymbolsX_Com1.', M,  'OutputType', 'bit', 'UnitAveragePower', true));
        Tx_BitsX=transpose(qamdemod(Trans.', M,  'OutputType', 'bit', 'UnitAveragePower', true));%, 'NoiseVariance', Norm_Noise_PowerX_Mes); %M QAM soft demod

        %% BER calculations
        Frame_ErrorsX = sum(Tx_BitsX(COI_no, 101:end-100)~=Rx_BitsX(101:end-100));  % Errors in a loop (x code blocks)

%% For only X-polarization BER calculation
        Total_Errors = Total_Errors+Frame_ErrorsX;
        Total_Bits = Total_Bits+length(Rx_BitsX)-200;
              
        loop_count = loop_count+1; % just to control loop during testing
        if ((Total_Errors>1000 && Total_Bits>=10^5) || Total_Bits>=10^6 || loop_count == 1) %ber accuracy control
            bool = false;
        end
   BER = Total_Errors/Total_Bits;
   Q = 20*log10(sqrt(2)*erfcinv(2*BER));
   
   BER_min=[BER_min BER];
   Q_max=[Q_max Q];
   Q_max
   
end
BER_min=min(BER_min)
Q_max=max(Q_max)