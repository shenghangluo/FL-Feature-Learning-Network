clear all
close all
clc
N_symb=32768;

disp('Please enter the launch power:')
n = input('[-4 -3 -2 -1 0 1 2 3 4 5]:');

switch n
    
 case -4
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\DATASET\VALIDATION_AND_TESTING_DATA\Dataset_n4_test.mat')
alpha=10^(0.1*(-4-5));
 case -3
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\DATASET\VALIDATION_AND_TESTING_DATA\Dataset_n3_test.mat')
alpha=10^(0.1*(-3-5));
 case -2
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\DATASET\VALIDATION_AND_TESTING_DATA\Dataset_n2_test.mat')
alpha=10^(0.1*(-2-5));
 case -1
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\DATASET\VALIDATION_AND_TESTING_DATA\Dataset_n1_test.mat')
alpha=10^(0.1*(-1-5));
 case 0
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\DATASET\VALIDATION_AND_TESTING_DATA\Dataset_0_test.mat')
alpha=10^(0.1*(0-5));
 case 1
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\DATASET\VALIDATION_AND_TESTING_DATA\Dataset_1_test.mat')
alpha=10^(0.1*(1-5));
 case 2
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\DATASET\VALIDATION_AND_TESTING_DATA\Dataset_2_test.mat')
alpha=10^(0.1*(2-5));
 case 3
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\DATASET\VALIDATION_AND_TESTING_DATA\Dataset_3_test.mat')
alpha=10^(0.1*(3-5));
 case 4
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\DATASET\VALIDATION_AND_TESTING_DATA\Dataset_4_test.mat')
alpha=10^(0.1*(4-5));
 case 5
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\DATASET\VALIDATION_AND_TESTING_DATA\Dataset_5_test.mat')
alpha=10^(0.1*(5-5));
end

disp('DNN-FO/SO-PB-NLC without pruning/PCA or with pruning and/or PCA?')
n = input('Type 0 for without pruning/PCA and 1 for with pruning and/or PCA');

switch n

case 0
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\TENSORFLOW_CODE\DNN_SO_PB_NLC\FO_Nonlinear_Distortion_Field_Real.csv')
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\TENSORFLOW_CODE\DNN_SO_PB_NLC\FO_Nonlinear_Distortion_Field_Imag.csv')
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\TENSORFLOW_CODE\DNN_SO_PB_NLC\SO_Nonlinear_Distortion_Field_Real.csv')
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\TENSORFLOW_CODE\DNN_SO_PB_NLC\SO_Nonlinear_Distortion_Field_Imag.csv')

case 1
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\TENSORFLOW_CODE\DNN_SO_PB_NLC_WITH_PCA_AND_PRUNING\FO_Nonlinear_Distortion_Field_Real.csv')
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\TENSORFLOW_CODE\DNN_SO_PB_NLC_WITH_PCA_AND_PRUNING\FO_Nonlinear_Distortion_Field_Imag.csv')
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\TENSORFLOW_CODE\DNN_SO_PB_NLC_WITH_PCA_AND_PRUNING\SO_Nonlinear_Distortion_Field_Real.csv')
load('C:\Sunish\Post_Doc_Work_UBC_2020\DNN_SO_PB_NLC_CODEBASE\TENSORFLOW_CODE\DNN_SO_PB_NLC_WITH_PCA_AND_PRUNING\SO_Nonlinear_Distortion_Field_Imag.csv')
end

datareal_FO=FO_Nonlinear_Distortion_Field_Real;
dataimage_FO=FO_Nonlinear_Distortion_Field_Imag;

datareal_SO=SO_Nonlinear_Distortion_Field_Real;
dataimage_SO=SO_Nonlinear_Distortion_Field_Imag;

COI_no=1;
M=64;
Total_Errors = 0;
loop_count = 0;
Total_Bits = 0;
pow_loop = 1;



 BER_min=[];
 Q_max=[];

%%-----------------------------------------------------------------------------------
complexdata_FO = complex(datareal_FO, dataimage_FO);
complexdata_FO = reshape(complexdata_FO, [1,N_symb]);

complexdata_SO = complex(datareal_SO, dataimage_SO);
complexdata_SO = reshape(complexdata_SO, [1,N_symb]);

disp('Is DNN-FO-PB-NLC or DNN-SO-PB-NLC?')
n = input('0: DNN-FO-PB-NLC and 1: DNN-SO-PB-NLC:');

switch n 
 case 0
Rx_SymbolsX_Com1 = Rx_SymbolsX_Com-(alpha.*complexdata_FO);
 case 1
Rx_SymbolsX_Com1 = Rx_SymbolsX_Com-(alpha.*complexdata_SO);

end

scatterplot(Rx_SymbolsX_Com1)
        %% QAM Demapper
        Rx_BitsX=transpose(qamdemod(Rx_SymbolsX_Com1.', M,  'OutputType', 'bit', 'UnitAveragePower', true));%, 'NoiseVariance', Norm_Noise_PowerX_Mes); %M QAM soft demod
        Rx_BitsY=transpose(qamdemod(Rx_SymbolsY_Com.', M,  'OutputType', 'bit', 'UnitAveragePower', true));%, 'NoiseVariance', Norm_Noise_PowerY_Mes);

        %% BER calculations

        Frame_ErrorsX = sum(Tx_BitsX(COI_no, 101:end-100)~=Rx_BitsX(101:end-100));  % Errors in a loop (x code blocks)
        Frame_ErrorsY = sum(Tx_BitsY(COI_no, 101:end-100)~=Rx_BitsY(101:end-100));


%% For only X-polarization BER calculation

        Total_Errors = Total_Errors+Frame_ErrorsX;
        Total_Bits = Total_Bits+length(Rx_BitsX)-400;
              
        loop_count = loop_count+1; % just to control loop during testing
        if ((Total_Errors>1000 && Total_Bits>=10^5) || Total_Bits>=10^6 || loop_count == 1) %ber accuracy control
            bool = false;
        end
        disp(['Running ', num2str(pow_loop), ' out of ', num2str(length(Tx_Pow_dBm)), ' main loop with sub loop ', num2str(loop_count)]);    

 
   BER = Total_Errors/Total_Bits;
   Q = 20*log10(sqrt(2)*erfcinv(2*BER));
   
   BER_min=[BER_min BER];
   Q_max=[Q_max Q];
   Q_max

BER_min=min(BER_min)
Q_max=max(Q_max)
