
clear all;
close all;
clc;
c = 3e8;                                % Speed of light in the vacuum
e = 1.6e-19;                            % Electron charge
h = 6.626068e-34;  
lambda0  = 1552.93; 
% lambda0  = 1550; 
L=4000e3;
Lspan=80e3;
% f_s=5.3750*10^9;
f_s=32*10^9;
Ts=1/f_s;
nFFT=4096;
Chrom_Disp=17;
n2 = 1.2e-20;                           % Kerr non-linear parameter
Seff = 80.0; 
alpha_dB=0.2;
alpha = alpha_dB * log(10)/10 * 1e-3;
% alpha = alpha_dB * log(10)/10;

D = (Chrom_Disp*1e-6);                            % Chromatic dispersion in s/m/m

beta2 = (lambda0*1e-9)^2/(2*pi*c) * D;
Gamma = 2*pi*n2/(lambda0*1e-9)/(Seff*1e-12);
% Gamma=Gamma*10^3;
dLeff = ((1 - exp(-alpha*Lspan))/alpha);
Leff = ((1 - exp(-alpha*L))/alpha);
% delta_f=f_s/nFFT;
tto=(Ts)/1;
N=200;

Threshold=-40;

k=0;
sample_m=[-N/2:1:N/2];
sample_n=[-N/2:1:N/2];
for i=1:1:N+1
   m=sample_m(i);
    if m~=0
    for j=1:1:N+1
        n=sample_n(j);
        if n~=0

  NL_coeff(i,j)=((1i*(8/9)*Gamma*tto^2)/(sqrt(3)*abs(beta2))*expint((-1i*m*n*Ts^2)/(beta2*L)));
       else
            
         NL_coeff(i,j)=((1i*(8/9)*Gamma*tto^2)/(sqrt(3)*abs(beta2))*(1/2)*expint(((n-m)^2*Ts^2*tto^2)/(3*abs(beta2)^2*L^2)));   
        end
        
    end
    else
       for j=1:1:N+1
        n=sample_n(j);
        if n~=0 
    NL_coeff(i,j)=((1i*(8/9)*Gamma*tto^2)/(sqrt(3)*abs(beta2))*(1/2)*expint(((n-m)^2*Ts^2*tto^2)/(3*abs(beta2)^2*L^2)));
        else
            q=nonlinear_coeff_zero_zero(tto,beta2,L);
          NL_coeff(i,j)=(1i*(8/9)*Gamma*tto^2)/(sqrt(3)*abs(beta2))*q;  

        end
        
       end
    end
    
end
NL_coeff=(dLeff/Lspan).*NL_coeff;
% NL_coeff_fiber=NL_coeff;
% save NL_coeff_fiber;
% NL_coeff(N/2+1,N/2+1)=0;
save NL_coeff;
NL_coeff_centre=NL_coeff(N/2+1,N/2+1);
NL_coeff(:,N/2+1)=[];
NL_coeff(N/2+1,:)=[];
s_coeff=size(NL_coeff);
NL_coeff_trunc=NL_coeff;
m_value=0;
n_value=0;
count=0;
count1=0;


for k=1:1:N
    for l=1:1:N
        coeff_temp=NL_coeff(k,l);
        coeff_thresh=20*log10(abs(coeff_temp)/abs(NL_coeff_centre));
        if floor(coeff_thresh)>Threshold
        count=count+1;
        else
          NL_coeff_trunc(k,l)=0;  
        end
    end
end
count
%  size(NL_coeff_new)     
% save NL_coeff;

S=size(NL_coeff_trunc);
NL_coeff_trunc=reshape(NL_coeff_trunc,1,S(1)*S(2));
leng=length(NL_coeff_trunc);
NL_coeff_trunc_index=[];
for o=1:1:leng
    if NL_coeff_trunc(o)==0
        NL_coeff_trunc_index=[NL_coeff_trunc_index o];
    end
end
% NL_coeff_trunc_index
NL_coeff_trunc(NL_coeff_trunc_index)=[];
save NL_coeff_trunc;
save NL_coeff_trunc_index;
%%Contour Plot
figure;
set(0,'DefaultAxesFontSize',28)
X_data=[-N/2:1:N/2-1];
Y_data=[-N/2:1:N/2-1];
% A=(NL_coeff)/(NL_coeff(N/2+1,N/2+1));
Q=(abs(NL_coeff)/abs(NL_coeff_centre));
% Q_thre=20*log10(abs(NL_coeff(1,1))/abs(NL_coeff(N/2+1,N/2+1)))

Q=(20*log10(Q));
  Q=Q.';
  
  [X,Y] = meshgrid(X_data,Y_data);
  contourf(X,Y,Q,[-10:-10:-50],'ShowText','on');grid on;
 
  axis([-N/2-10 N/2+10 -N/2-10 N/2+10])
% set(gca,'XTick',-6:1:3)
% set(gca,'YTick',4:1:8)
set(0,'DefaultAxesFontName', 'Times New Roman')
% set(0,'DefaultAxesFontSize',28)
  xlabel('Pulse index m')
 ylabel('Pulse index n');
%   title('D.64-QAM-OFDM');
