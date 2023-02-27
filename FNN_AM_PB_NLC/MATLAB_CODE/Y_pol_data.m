clear all
clc
load('G:\MASc\Perturbation_based\learned_A_M_Model\Dataset\Dataset_Dual_test_n1_PB.mat')
M = 64;
Tx_SymbolsY(1, :) = transpose(qammod(Tx_BitsY(1, :).', M,  'InputType', 'bit', 'UnitAveragePower', true));   %M QAM modulation

Transmit_real_Y = real(Tx_SymbolsY);
Transmit_image_Y = imag(Tx_SymbolsY);

Input_Symbol_real_Y = real(Rx_SymbolsY_Com);
Input_Symbol_image_Y = imag(Rx_SymbolsY_Com);

save('G:\MASc\Perturbation_based\learned_A_M_Model\Dataset\XY_Dataset_Dual_test_n1_PB.mat','Tx_BitsX','Tx_BitsY','Rx_SymbolsX_Com','Rx_SymbolsY_Com', 'Tx_Pow_dBm', 'Input_Symbol_real_X','Input_Symbol_image_X','Transmit_real_X','Transmit_image_X', 'Input_Symbol_real_Y','Input_Symbol_image_Y','Transmit_real_Y','Transmit_image_Y');         


