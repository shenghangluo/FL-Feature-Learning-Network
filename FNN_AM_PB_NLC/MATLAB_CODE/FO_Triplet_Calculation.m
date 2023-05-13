
close all
clear all
clc;
perturb_symbol_legth=100;
Parameters_for_FO_PB_NLC(perturb_symbol_legth);

load('Dataset_0_train.mat');
% load('Dataset_6_test.mat');


X_data=Rx_SymbolsX_Com;
Y_data=Rx_SymbolsY_Com;

load A;
load B;
load C;

% Peak_power = 10^((Peak_power_dBm-30)/10);

symb_total_legth=length(X_data);
X_data_temp=[zeros(1,symb_total_legth) X_data zeros(1,symb_total_legth)];
Y_data_temp=[zeros(1,symb_total_legth) Y_data zeros(1,symb_total_legth)];

leng=length(A);

Triplets=zeros(symb_total_legth,leng);

   
for symb=1:1:symb_total_legth

    Triplets(symb,:)=((((X_data_temp(symb_total_legth+symb+A).*conj(X_data_temp(symb_total_legth+symb+C))...
        +Y_data_temp(symb_total_legth+symb+A).*conj(Y_data_temp(symb_total_legth+symb+C)))...
        .*X_data_temp(symb_total_legth+symb+B))));

end

Trip_real_X=real(Triplets);
Trip_image_X=imag(Triplets);

% Triplets=((epslon*Peak_power)^(3/2)).*Triplet;
save('Trip_0_train_FO.mat', 'Trip_real_X','Trip_image_X');
% save('Trip_6_test_FO.mat', 'Trip_real_X','Trip_image_X');        
