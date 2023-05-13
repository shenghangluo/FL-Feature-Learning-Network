load NL_coeff_second_order_PART2_NLSE_3D;
NL_coeff_second_order_PART2_NLSE_3D_trunc=NL_coeff_second_order_PART2_NLSE_3D;
ParametersWDM;
N=perturb_symbol_legth;
NL_coeff_centre=NL_coeff_second_order_PART2_NLSE_3D(N/2+1,N/2+1,N/2+1);

Threshold=-35;
count=0;
for i=1:1:N+1
    for j=1:1:N+1
        for k=1:1:N+1
            
        coeff_temp=NL_coeff_second_order_PART2_NLSE_3D_trunc(i,j,k);
        coeff_thresh=20*log10(abs(coeff_temp)/abs(NL_coeff_centre));
        if floor(coeff_thresh)>Threshold
        count=count+1;
        else
           NL_coeff_second_order_PART2_NLSE_3D_trunc(i,j,k)=0; 
        end
            end
            
            end
end 
count

NL_coeff_second_order_PART2_NLSE_3D_trunc(N/2+1,N/2+1,:)=0;
NL_coeff_second_order_PART2_NLSE_3D_trunc(:,N/2+1,N/2+1)=0;
NL_coeff_second_order_PART2_NLSE_3D_trunc(N/2+1,:,N/2+1)=0;

S=size(NL_coeff_second_order_PART2_NLSE_3D_trunc);
NL_coeff_second_order_PART2_NLSE_3D_trunc=reshape(NL_coeff_second_order_PART2_NLSE_3D_trunc,1,S(1)*S(2)*S(3));
leng=length(NL_coeff_second_order_PART2_NLSE_3D_trunc)
NL_coeff_second_order_PART2_NLSE_3D_trunc_index=[];
for o=1:1:leng
    if NL_coeff_second_order_PART2_NLSE_3D_trunc(o)==0
        NL_coeff_second_order_PART2_NLSE_3D_trunc_index=[NL_coeff_second_order_PART2_NLSE_3D_trunc_index o];
    end
end
  size(NL_coeff_second_order_PART2_NLSE_3D_trunc_index)
  NL_coeff_second_order_PART2_NLSE_3D_trunc(NL_coeff_second_order_PART2_NLSE_3D_trunc_index)=[];
  save NL_coeff_second_order_PART2_NLSE_3D_trunc;
  save NL_coeff_second_order_PART2_NLSE_3D_trunc_index;
  