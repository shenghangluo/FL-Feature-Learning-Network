clear all
clc

load('Dataset_16QAM_1000_5WDM32G_PreCDC_train_2.mat')
X_data=Rx_SymbolsX_Com;
Y_data=Rx_SymbolsY_Com;
load('FWM_Triplets_16QAM_1000_PreCDC_5WDM32G_75_train_2.mat')
Trip_real_X_IFWM = Trip_real_X;
Trip_image_X_IFWM = Trip_image_X;
p=1.25;

L=75;
mnInd = {};
ii=0;
clear min
min = min([p*ceil(L/2)/abs(ii)], [floor(L/2)]);
min = floor(min);
ii_n = -min:1:min;

mnInd{1,ii+1}=ii_n(ii_n~=0);

sum(~cellfun(@isempty,mnInd))
sum(cellfun(@numel,mnInd))

symb_total_legth=length(X_data);
X_data_temp=[zeros(1,symb_total_legth) X_data zeros(1,symb_total_legth)];
Y_data_temp=[zeros(1,symb_total_legth) Y_data zeros(1,symb_total_legth)];
for symb=1:1:symb_total_legth
    Triplets_row=[];
         m=0;
         n=mnInd{1,m+1};

         Triplets_1=Y_data_temp(symb_total_legth+symb).*conj(Y_data_temp(symb_total_legth+symb+n)).*X_data_temp(symb_total_legth+symb+n);
         Triplets_row=[Triplets_row, Triplets_1];

     Triplets(symb,:)=Triplets_row;
end

ICIXPM_Trip_real_X=real(Triplets);
ICIXPM_Trip_image_X=imag(Triplets);

Trip_real_X = [Trip_real_X_IFWM,ICIXPM_Trip_real_X];
Trip_image_X = [Trip_image_X_IFWM,ICIXPM_Trip_image_X];
size(Trip_real_X)
save('ADD_Triplets_16QAM_1000_PreCDC_5WDM32G_75_train_2.mat', 'Trip_real_X','Trip_image_X');

clear all
clc
load('Dataset_16QAM_1000_5WDM32G_PreCDC_test_2.mat') 
X_data=Rx_SymbolsX_Com;
Y_data=Rx_SymbolsY_Com;
load('FWM_Triplets_16QAM_1000_PreCDC_5WDM32G_75_test_2.mat')
Trip_real_X_IFWM = Trip_real_X;
Trip_image_X_IFWM = Trip_image_X;
p=1.25;
L=75;
mnInd = {};
ii=0;
clear min
min = min([p*ceil(L/2)/abs(ii)], [floor(L/2)]);
min = floor(min);
ii_n = -min:1:min;

mnInd{1,ii+1}=ii_n(ii_n~=0);

sum(~cellfun(@isempty,mnInd))
sum(cellfun(@numel,mnInd))

symb_total_legth=length(X_data);
X_data_temp=[zeros(1,symb_total_legth) X_data zeros(1,symb_total_legth)];
Y_data_temp=[zeros(1,symb_total_legth) Y_data zeros(1,symb_total_legth)];
for symb=1:1:symb_total_legth
    Triplets_row=[];
         m=0;
         n=mnInd{1,m+1};

         Triplets_1=Y_data_temp(symb_total_legth+symb).*conj(Y_data_temp(symb_total_legth+symb+n)).*X_data_temp(symb_total_legth+symb+n);
         Triplets_row=[Triplets_row, Triplets_1];

     Triplets(symb,:)=Triplets_row;
end

ICIXPM_Trip_real_X=real(Triplets);
ICIXPM_Trip_image_X=imag(Triplets);

Trip_real_X = [Trip_real_X_IFWM,ICIXPM_Trip_real_X];
Trip_image_X = [Trip_image_X_IFWM,ICIXPM_Trip_image_X];
size(Trip_real_X)

save('ADD_Triplets_16QAM_1000_PreCDC_5WDM32G_75_test_2.mat', 'Trip_real_X','Trip_image_X');

clear all
clc

load('Dataset_16QAM_1000_5WDM32G_PreCDC_train_2.mat')
X_data=Rx_SymbolsX_Com;
Y_data=Rx_SymbolsY_Com;
p=1.25;
L=75;
mnInd = {};
ii=0;
clear min
min = min([p*ceil(L/2)/abs(ii)], [floor(L/2)]);
min = floor(min);
ii_n = -min:1:min;

mnInd{1,ii+1}=ii_n(ii_n~=0);

sum(~cellfun(@isempty,mnInd))
sum(cellfun(@numel,mnInd))

symb_total_legth=length(X_data);
X_data_temp=[zeros(1,symb_total_legth) X_data zeros(1,symb_total_legth)];
Y_data_temp=[zeros(1,symb_total_legth) Y_data zeros(1,symb_total_legth)];
for symb=1:1:symb_total_legth
    Triplets_row=[];
         m=0;
         n=mnInd{1,m+1};

         Triplets_1=2*X_data_temp(symb_total_legth+symb+n).*conj(X_data_temp(symb_total_legth+symb+n))+Y_data_temp(symb_total_legth+symb+n).*conj(Y_data_temp(symb_total_legth+symb+n));
         Triplets_row=[Triplets_row, Triplets_1];

     Triplets(symb,:)=Triplets_row;
end

Multi_Trip_real_X=real(Triplets);
size(Multi_Trip_real_X)
save('IXPM_Triplets_16QAM_1000_PreCDC_5WDM32G_75_train_2.mat', 'Multi_Trip_real_X');

clear all
clc

load('Dataset_16QAM_1000_5WDM32G_PreCDC_test_2.mat') 
X_data=Rx_SymbolsX_Com;
Y_data=Rx_SymbolsY_Com;

p=1.25;

L=75;
mnInd = {};
ii=0;
clear min
min = min([p*ceil(L/2)/abs(ii)], [floor(L/2)]);
min = floor(min);
ii_n = -min:1:min;

mnInd{1,ii+1}=ii_n(ii_n~=0);

sum(~cellfun(@isempty,mnInd))
sum(cellfun(@numel,mnInd))

symb_total_legth=length(X_data);
X_data_temp=[zeros(1,symb_total_legth) X_data zeros(1,symb_total_legth)];
Y_data_temp=[zeros(1,symb_total_legth) Y_data zeros(1,symb_total_legth)];
for symb=1:1:symb_total_legth
    Triplets_row=[];
         m=0;
         n=mnInd{1,m+1};

         Triplets_1=2*X_data_temp(symb_total_legth+symb+n).*conj(X_data_temp(symb_total_legth+symb+n))+Y_data_temp(symb_total_legth+symb+n).*conj(Y_data_temp(symb_total_legth+symb+n));
         Triplets_row=[Triplets_row, Triplets_1];

     Triplets(symb,:)=Triplets_row;
end

Multi_Trip_real_X=real(Triplets);
size(Multi_Trip_real_X)
save('IXPM_Triplets_16QAM_1000_PreCDC_5WDM32G_75_test_2.mat', 'Multi_Trip_real_X');

clear all
clc

load('Dataset_16QAM_1000_5WDM32G_PreCDC_train_2.mat')
X_data=Rx_SymbolsX_Com;
Y_data=Rx_SymbolsY_Com;
load('IXPM_Triplets_16QAM_1000_PreCDC_5WDM32G_75_train_2.mat')
p=1.25;
symb_total_legth=length(X_data);
X_data_temp=[zeros(1,symb_total_legth) X_data zeros(1,symb_total_legth)];
Y_data_temp=[zeros(1,symb_total_legth) Y_data zeros(1,symb_total_legth)];
for symb=1:1:symb_total_legth
    Triplets_row=[];
         m=0;
         n=0;

         Triplets_1=X_data_temp(symb_total_legth+symb+n).*conj(X_data_temp(symb_total_legth+symb+n))+Y_data_temp(symb_total_legth+symb+n).*conj(Y_data_temp(symb_total_legth+symb+n));
         Triplets_row=[Triplets_row, Triplets_1];

     Triplets(symb,:)=Triplets_row;
end
Multi_Trip_real_X=[Multi_Trip_real_X, real(Triplets)];
size(Multi_Trip_real_X)                                               
save('Mul_Triplets_16QAM_1000_PreCDC_5WDM32G_75_train_2.mat', 'Multi_Trip_real_X'); 

clear all
clc

load('Dataset_16QAM_1000_5WDM32G_PreCDC_test_2.mat') 
X_data=Rx_SymbolsX_Com;
Y_data=Rx_SymbolsY_Com;
load('IXPM_Triplets_16QAM_1000_PreCDC_5WDM32G_75_test_2.mat')
p=1.25;
symb_total_legth=length(X_data);
X_data_temp=[zeros(1,symb_total_legth) X_data zeros(1,symb_total_legth)];
Y_data_temp=[zeros(1,symb_total_legth) Y_data zeros(1,symb_total_legth)];
for symb=1:1:symb_total_legth
    Triplets_row=[];
         m=0;
         n=0;

         Triplets_1=X_data_temp(symb_total_legth+symb+n).*conj(X_data_temp(symb_total_legth+symb+n))+Y_data_temp(symb_total_legth+symb+n).*conj(Y_data_temp(symb_total_legth+symb+n));
         Triplets_row=[Triplets_row, Triplets_1];

     Triplets(symb,:)=Triplets_row;
end

Multi_Trip_real_X=[Multi_Trip_real_X, real(Triplets)];

size(Multi_Trip_real_X)
save('Mul_Triplets_16QAM_1000_PreCDC_5WDM32G_75_test_2.mat', 'Multi_Trip_real_X'); 