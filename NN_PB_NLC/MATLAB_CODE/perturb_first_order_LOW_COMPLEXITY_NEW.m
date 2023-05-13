function delta_x=perturb_first_order_LOW_COMPLEXITY_NEW(X_data,Y_data,Peak_power_dBm,epslon)
clc;
load NL_coeff_trunc;
load A;
load B;
load C;
Peak_power = 10^((Peak_power_dBm-30)/10);
symb_total_legth=length(X_data);
X_data_temp=[zeros(1,symb_total_legth) X_data zeros(1,symb_total_legth)];
Y_data_temp=[zeros(1,symb_total_legth) Y_data zeros(1,symb_total_legth)];

delta_x=[zeros(1,symb_total_legth)];

for symb=1:1:symb_total_legth
%     delta_x(symb)=sum(sum(X_data_temp(symb_total_legth+symb+A).*X_data_temp(symb_total_legth+symb+B).*...
%         conj(X_data_temp(symb_total_legth+symb+C)).*NL_coeff_trunc));
%     
    delta_x(symb)=sum(sum(((X_data_temp(symb_total_legth+symb+A).*conj(X_data_temp(symb_total_legth+symb+C))...
        +Y_data_temp(symb_total_legth+symb+A).*conj(Y_data_temp(symb_total_legth+symb+C)))...
        .*X_data_temp(symb_total_legth+symb+B)).*NL_coeff_trunc));
end

delta_x=((epslon*Peak_power)^(3/2)).*delta_x;
end
      
        
