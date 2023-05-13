clear all
close all
clc


load('Dataset_16QAM_1000_5WDM32G_PreCDC_test_2.mat')
Half_Window_size = 12;
X_real = [zeros(1,Half_Window_size),Input_Symbol_real_X, zeros(1,Half_Window_size)];
X_image = [zeros(1,Half_Window_size),Input_Symbol_image_X, zeros(1,Half_Window_size)];
save('Prossed_long_sequence_Dataset_16QAM_1000_PreCDC_5WDM32G_X_test_2.mat', 'X_real', 'X_image', 'Input_Symbol_real_X', 'Input_Symbol_image_X', 'Transmit_real_X', 'Transmit_image_X', 'Tx_BitsX', 'Rx_SymbolsX_Com')


load('Dataset_16QAM_1000_5WDM32G_PreCDC_test_2.mat')
Half_Window_size = 12;
Y_real = [zeros(1,Half_Window_size),Input_Symbol_real_Y, zeros(1,Half_Window_size)];
Y_image = [zeros(1,Half_Window_size),Input_Symbol_image_Y, zeros(1,Half_Window_size)];
save('Prossed_long_sequence_Dataset_16QAM_1000_PreCDC_5WDM32G_Y_test_2.mat', 'Y_real', 'Y_image', 'Input_Symbol_real_Y', 'Input_Symbol_image_Y', 'Transmit_real_Y', 'Transmit_image_Y', 'Tx_BitsY', 'Rx_SymbolsY_Com')
