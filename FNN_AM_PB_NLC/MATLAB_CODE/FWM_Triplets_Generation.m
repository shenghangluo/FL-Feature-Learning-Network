clear all
close all
clc

load('Dataset_16QAM_1000_5WDM32G_PreCDC_train_2.mat')
X_data=Rx_SymbolsX_Com;
Y_data=Rx_SymbolsY_Com;
p=1.25;
L=75;
mnInd = {};
for ii=1:1:ceil(L/2)
    m = ii-1;
     clear min
     min = min([p*ceil(L/2)/abs(m)], [floor(L/2)]);
     min = floor(min);
     ii_n = -min:1:min;
     ii_n = ii_n(ii_n~=0);
     mnInd{1,ii}=ii_n;
end

sum(~cellfun(@isempty,mnInd))
sum(cellfun(@numel,mnInd))*2-L+1

symb_total_legth=length(X_data);
X_data_temp=[zeros(1,symb_total_legth) X_data zeros(1,symb_total_legth)];
Y_data_temp=[zeros(1,symb_total_legth) Y_data zeros(1,symb_total_legth)];


for symb=1:1:symb_total_legth
    Triplets_row=[];
    for ii=1:1:ceil(L/2)
         m=ii-1;
         n=mnInd{1,ii};
         if m~=0 
            Triplets_1=((((X_data_temp(symb_total_legth+symb+m).*conj(X_data_temp(symb_total_legth+symb+(m+n)))...
                        +Y_data_temp(symb_total_legth+symb+m).*conj(Y_data_temp(symb_total_legth+symb+(m+n))))...
                        .*X_data_temp(symb_total_legth+symb+n))));
                if m~=0
                    Triplets_2=((((X_data_temp(symb_total_legth+symb+(-m)).*conj(X_data_temp(symb_total_legth+symb+(-m+n)))...
                        +Y_data_temp(symb_total_legth+symb+(-m)).*conj(Y_data_temp(symb_total_legth+symb+(-m+n))))...
                        .*X_data_temp(symb_total_legth+symb+n))));
                    
                    temp = Triplets_1+Triplets_2;
                else
                    temp = Triplets_1;
                end
            Triplets_row=[Triplets_row, temp];
        end
    end

     Triplets(symb,:)=Triplets_row;
end

Trip_real_X=real(Triplets);
Trip_image_X=imag(Triplets);

size(Trip_real_X)

save('FWM_Triplets_16QAM_1000_PreCDC_5WDM32G_75_train_2.mat', 'Trip_real_X','Trip_image_X');

clear all
clc

load('Dataset_16QAM_1000_5WDM32G_PreCDC_test_2.mat')
X_data=Rx_SymbolsX_Com;
Y_data=Rx_SymbolsY_Com;
p=1.25;
L=75;
mnInd = {};
for ii=1:1:ceil(L/2)
    m = ii-1;
     clear min
     min = min([p*ceil(L/2)/abs(m)], [floor(L/2)]);
     min = floor(min);
     ii_n = -min:1:min;
     ii_n = ii_n(ii_n~=0);
     mnInd{1,ii}=ii_n;
end

sum(~cellfun(@isempty,mnInd))
sum(cellfun(@numel,mnInd))*2-L+1

symb_total_legth=length(X_data);
X_data_temp=[zeros(1,symb_total_legth) X_data zeros(1,symb_total_legth)];
Y_data_temp=[zeros(1,symb_total_legth) Y_data zeros(1,symb_total_legth)];


for symb=1:1:symb_total_legth
    Triplets_row=[];
    for ii=1:1:ceil(L/2)
         m=ii-1;
         n=mnInd{1,ii};
         if m~=0 
            Triplets_1=((((X_data_temp(symb_total_legth+symb+m).*conj(X_data_temp(symb_total_legth+symb+(m+n)))...
                        +Y_data_temp(symb_total_legth+symb+m).*conj(Y_data_temp(symb_total_legth+symb+(m+n))))...
                        .*X_data_temp(symb_total_legth+symb+n))));
                if m~=0
                    Triplets_2=((((X_data_temp(symb_total_legth+symb+(-m)).*conj(X_data_temp(symb_total_legth+symb+(-m+n)))...
                        +Y_data_temp(symb_total_legth+symb+(-m)).*conj(Y_data_temp(symb_total_legth+symb+(-m+n))))...
                        .*X_data_temp(symb_total_legth+symb+n))));
                    
                    temp = Triplets_1+Triplets_2;
                else
                    temp = Triplets_1;
                end
            Triplets_row=[Triplets_row, temp];
        end
    end

     Triplets(symb,:)=Triplets_row;
end

Trip_real_X=real(Triplets);
Trip_image_X=imag(Triplets);

size(Trip_real_X)

save('FWM_Triplets_16QAM_1000_PreCDC_5WDM32G_75_test_2.mat', 'Trip_real_X','Trip_image_X');