clear all
close all
clc
load('Dataset_16QAM_1000_5WDM32G_PreCDC_test_2.mat')
N_Transient = 100;
Window_size = 25;
N_tot = 2^16;
input_real = zeros(N_tot, 2*N_Transient+Window_size*2+1);
%%%%%%%%%% Input_Real
for ii = 1:N_tot
input_real(ii,N_Transient+Window_size+1)=Input_Symbol_real_X(1,ii);
end

for ii = 1:N_tot
    for jj = 1:Window_size
        if jj+ii<=N_tot
        input_real(ii, N_Transient+Window_size+1+jj) = Input_Symbol_real_X(1, ii+jj);
        else 
        input_real(ii, N_Transient+Window_size+1+jj) = 0;
        end
    end
end

for ii = 1:N_tot
    for jj = 1:Window_size
        if ii-Window_size-1+jj>=1
        input_real(ii, jj+N_Transient) = Input_Symbol_real_X(1, ii-Window_size-1+jj);
        else 
        input_real(ii, jj+N_Transient) = 0;
        end
    end
end

for ii = 1:N_tot
    for jj = 1:N_Transient
        if ii-Window_size-N_Transient-1+jj>=1
        input_real(ii, jj) = Input_Symbol_real_X(1, ii-Window_size-N_Transient-1+jj);
        else 
        input_real(ii, jj) = 0;
        end
    end
end

for ii = 1:N_tot
    for jj = 1:N_Transient
        if ii+Window_size+jj<=N_tot
        input_real(ii, N_Transient+Window_size*2+1+jj) = Input_Symbol_real_X(1, ii+Window_size+jj);
        else 
        input_real(ii, jj) = 0;
        end
    end
end

X_real = input_real;

%%%%%%%%%%%%%%%%%%%% Input_Image
input_image = zeros(N_tot,  2*N_Transient+Window_size*2+1);
for ii = 1:N_tot
input_image(ii,N_Transient+Window_size+1)=Input_Symbol_image_X(1,ii);
end

for ii = 1:N_tot
    for jj = 1:Window_size
        if jj+ii<=N_tot
        input_image(ii, N_Transient+Window_size+1+jj) = Input_Symbol_image_X(1, ii+jj);
        else 
        input_image(ii, N_Transient+Window_size+1+jj) = 0;
        end
    end
end

for ii = 1:N_tot
    for jj = 1:Window_size
        if ii-Window_size-1+jj>=1
        input_image(ii, jj+N_Transient) = Input_Symbol_image_X(1, ii-Window_size-1+jj);
        else 
        input_image(ii, jj+N_Transient) = 0;
        end
    end
end

for ii = 1:N_tot
    for jj = 1:N_Transient
        if ii-Window_size-N_Transient-1+jj>=1
        input_image(ii, jj) = Input_Symbol_image_X(1, ii-Window_size-N_Transient-1+jj);
        else 
        input_image(ii, jj) = 0;
        end
    end
end

for ii = 1:N_tot
    for jj = 1:N_Transient
        if ii+Window_size+jj<=N_tot
        input_image(ii, N_Transient+Window_size*2+1+jj) = Input_Symbol_image_X(1, ii+Window_size+jj);
        else 
        input_image(ii, jj) = 0;
        end
    end
end

X_image = input_image;
save('Prossed_X_Dataset_16QAM_1000_PreCDC_5WDM32G_test_2.mat', 'X_real', 'X_image', 'Input_Symbol_real_X', 'Input_Symbol_image_X', 'Transmit_real_X', 'Transmit_image_X', 'Tx_BitsX', 'Rx_SymbolsX_Com')
