clear all
close all
clc

load('Dataset_16QAM_1000_5WDM32G_PreCDC_test_2_all.mat')

M = 16;

Input_Symbol_real_X = Input_Symbol_real_X(6*2^15+1:8*2^15);
Input_Symbol_image_X = Input_Symbol_image_X(6*2^15+1:8*2^15);

Rx_SymbolsX_Com = Rx_SymbolsX_Com(6*2^15+1:8*2^15);
Rx_SymbolsY_Com = Rx_SymbolsY_Com(6*2^15+1:8*2^15);

Transmit_real_X = Transmit_real_X(6*2^15+1:8*2^15);
Transmit_image_X = Transmit_image_X(6*2^15+1:8*2^15);

Tx_BitsX = Tx_Bits_X(6*log2(M)*2^15+1:8*log2(M)*2^15);
Tx_BitsY = Tx_Bits_Y(6*log2(M)*2^15+1:8*log2(M)*2^15);

save('Dataset_16QAM_1000_5WDM32G_PreCDC_valid_2.mat')

clear all
close all
clc

load('Dataset_16QAM_1000_5WDM32G_PreCDC_test_2_all.mat')

M = 16;

Input_Symbol_real_X = Input_Symbol_real_X(8*2^15+1:end);
Input_Symbol_image_X = Input_Symbol_image_X(8*2^15+1:end);

Rx_SymbolsX_Com = Rx_SymbolsX_Com(8*2^15+1:end);
Rx_SymbolsY_Com = Rx_SymbolsY_Com(8*2^15+1:end);

Transmit_real_X = Transmit_real_X(8*2^15+1:end);
Transmit_image_X = Transmit_image_X(8*2^15+1:end);

Tx_BitsX = Tx_Bits_X(8*log2(M)*2^15+1:end);
Tx_BitsY = Tx_Bits_Y(8*log2(M)*2^15+1:end);

save('Dataset_16QAM_1000_5WDM32G_PreCDC_test_2.mat')
