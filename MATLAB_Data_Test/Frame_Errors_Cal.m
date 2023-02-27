function [Frame_Errors_CDC, Frame_Total_Bits] = Frame_Errors_Cal(Rx_SymbolsX_Com, Rx_SymbolsY_Com, M, Tx_BitsX_COI, Tx_BitsY_COI)

Rx_BitsX_CDC=transpose(qamdemod(Rx_SymbolsX_Com.', M,  'OutputType', 'bit', 'UnitAveragePower', true));%, 'NoiseVariance', Norm_Noise_PowerX_Mes); %M QAM soft demod
Rx_BitsY_CDC=transpose(qamdemod(Rx_SymbolsY_Com.', M,  'OutputType', 'bit', 'UnitAveragePower', true));%, 'NoiseVariance', Norm_Noise_PowerX_Mes); %M QAM soft demod

Frame_ErrorsX_CDC = sum(Tx_BitsX_COI(101:end-100)~=Rx_BitsX_CDC(101:end-100));  % Errors in a loop (x code blocks)
Frame_ErrorsY_CDC = sum(Tx_BitsY_COI(101:end-100)~=Rx_BitsY_CDC(101:end-100));  % Errors in a loop (x code blocks)

Frame_Errors_CDC = Frame_ErrorsX_CDC+Frame_ErrorsY_CDC;
Frame_Total_Bits = 2*(length(Rx_BitsX_CDC)-200);

end