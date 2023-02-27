load NL_coeff_second_order_PART1_NLSE_3D;
NL_coeff_second_order_PART1_NLSE_3D_trunc_2800Km=NL_coeff_second_order_PART1_NLSE_3D;
% ParametersWDM;
N=100;
NL_coeff_centre=NL_coeff_second_order_PART1_NLSE_3D(N/2+1,N/2+1,N/2+1);

Threshold=-20;
count=0;
for i=1:1:N+1
    for j=1:1:N+1
        for k=1:1:N+1
            
        coeff_temp=NL_coeff_second_order_PART1_NLSE_3D_trunc_2800Km(i,j,k);
        coeff_thresh=20*log10(abs(coeff_temp)/abs(NL_coeff_centre));
        if floor(coeff_thresh)>Threshold
        count=count+1;
        else
           NL_coeff_second_order_PART1_NLSE_3D_trunc_2800Km(i,j,k)=0; 
        end
            end
            
            end
end 
count

NL_coeff_second_order_PART1_NLSE_3D_trunc_2800Km(N/2+1,N/2+1,:)=0;
NL_coeff_second_order_PART1_NLSE_3D_trunc_2800Km(:,N/2+1,N/2+1)=0;
NL_coeff_second_order_PART1_NLSE_3D_trunc_2800Km(N/2+1,:,N/2+1)=0;

S=size(NL_coeff_second_order_PART1_NLSE_3D_trunc_2800Km);
NL_coeff_second_order_PART1_NLSE_3D_trunc_2800Km=reshape(NL_coeff_second_order_PART1_NLSE_3D_trunc_2800Km,1,S(1)*S(2)*S(3));
leng=length(NL_coeff_second_order_PART1_NLSE_3D_trunc_2800Km)
NL_coeff_second_order_PART1_NLSE_3D_trunc_index_2800Km=[];
for o=1:1:leng
    if NL_coeff_second_order_PART1_NLSE_3D_trunc_2800Km(o)==0
        NL_coeff_second_order_PART1_NLSE_3D_trunc_index_2800Km=[NL_coeff_second_order_PART1_NLSE_3D_trunc_index_2800Km o];
    end
end
  
  NL_coeff_second_order_PART1_NLSE_3D_trunc_2800Km(NL_coeff_second_order_PART1_NLSE_3D_trunc_index_2800Km)=[];
  save NL_coeff_second_order_PART1_NLSE_3D_trunc_2800Km;
  save NL_coeff_second_order_PART1_NLSE_3D_trunc_index_2800Km;
  size(NL_coeff_second_order_PART1_NLSE_3D_trunc_2800Km)
  