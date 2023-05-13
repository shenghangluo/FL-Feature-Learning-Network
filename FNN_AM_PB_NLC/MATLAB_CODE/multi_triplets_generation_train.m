clear all
clc

load('D:\MASC\Perturbation_based\learned_AM_Model\Dataset\Dataset_Dual_train_3.mat');
X_data=Rx_SymbolsX_Com;
Y_data=Rx_SymbolsY_Com;

p=2.5;

L=38;
mnInd = {};
ii=0;
clear min
min = min([p*ceil(L/2)/ii], [ceil(L/2)]);
min = ceil(min);
ii_n = -min:1:min;

mnInd{1,ii+1}=ii_n;

sum(~cellfun(@isempty,mnInd))
sum(cellfun(@numel,mnInd))

symb_total_legth=length(X_data);
X_data_temp=[zeros(1,symb_total_legth) X_data zeros(1,symb_total_legth)];
Y_data_temp=[zeros(1,symb_total_legth) Y_data zeros(1,symb_total_legth)];
for symb=1:1:symb_total_legth
    Triplets_row=[];
         m=0;
         n=mnInd{1,m+1};

         Triplets_1=X_data_temp(symb_total_legth+symb+n).*conj(X_data_temp(symb_total_legth+symb+n))+Y_data_temp(symb_total_legth+symb+n).*conj(Y_data_temp(symb_total_legth+symb+n));
        
         Triplets_row=[Triplets_row, Triplets_1];

     Triplets(symb,:)=Triplets_row;
end

Multi_Trip_real_X=real(Triplets);

size(Multi_Trip_real_X)

save('D:\MASC\Perturbation_based\learned_AM_Model\Dataset\Multi_Triplets_38_Dual_train_3.mat', 'Multi_Trip_real_X');
