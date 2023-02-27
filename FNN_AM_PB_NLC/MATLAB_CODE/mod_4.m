function [symbol] = mod_4(data1)

i = sqrt(-1);
l = 1;

for k=1:4:length(data1)
    if     (data1(k)==1 & data1(k+1)==1 & data1(k+2)==0 & data1(k+3)==0) % 1st quarter
          QAM16(l)=(1+3*i);  % field = A * e^(jf)...where A =amplitude=1 and f=phase
    elseif (data1(k)==1 & data1(k+1)==0 & data1(k+2)==0 & data1(k+3)==0)
          QAM16(l)=(3+3*i);   
    elseif (data1(k)==1 & data1(k+1)==1 & data1(k+2)==0 & data1(k+3)==1)
          QAM16(l)=(1+1*i);      
    elseif (data1(k)==1 & data1(k+1)==0 & data1(k+2)==0 & data1(k+3)==1)
          QAM16(l)=(3+1*i); 
    elseif (data1(k)==1 & data1(k+1)==1 & data1(k+2)==1 & data1(k+3)==1)% 4th quarter
          QAM16(l)=(1-1*i);
    elseif (data1(k)==1 & data1(k+1)==0 & data1(k+2)==1 & data1(k+3)==1)
          QAM16(l)=(3-1*i);
    elseif (data1(k)==1 & data1(k+1)==1 & data1(k+2)==1 & data1(k+3)==0)
          QAM16(l)=(1-3*i);
    elseif (data1(k)==1 & data1(k+1)==0 & data1(k+2)==1 & data1(k+3)==0)
          QAM16(l)=(3-3*i);
    elseif (data1(k)==0 & data1(k+1)==0 & data1(k+2)==0 & data1(k+3)==0)% 2nd quarter
          QAM16(l)=(-3+3*i);
    elseif (data1(k)==0 & data1(k+1)==1 & data1(k+2)==0 & data1(k+3)==0)
          QAM16(l)=(-1+3*i);
    elseif (data1(k)==0 & data1(k+1)==0 & data1(k+2)==0 & data1(k+3)==1)
          QAM16(l)=(-3+1*i);
    elseif (data1(k)==0 & data1(k+1)==1 & data1(k+2)==0 & data1(k+3)==1)
          QAM16(l)=(-1+1*i);
    elseif (data1(k)==0 & data1(k+1)==0 & data1(k+2)==1 & data1(k+3)==1)% 3rd quarter
          QAM16(l)=(-3-1*i);
    elseif (data1(k)==0 & data1(k+1)==1 & data1(k+2)==1 & data1(k+3)==1)
          QAM16(l)=(-1-1*i);
    elseif (data1(k)==0 & data1(k+1)==0 & data1(k+2)==1 & data1(k+3)==0)
          QAM16(l)=(-3-3*i);
    elseif (data1(k)==0 & data1(k+1)==1 & data1(k+2)==1 & data1(k+3)==0)
          QAM16(l)=(-1-3*i);   
    end
    l = l+1;
end

QAM16 = 1/(2*sqrt(2)) * QAM16;                  % 16QAM power Normalisation to 1

symbol=QAM16;
