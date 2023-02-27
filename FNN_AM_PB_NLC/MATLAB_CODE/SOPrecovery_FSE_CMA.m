function f = SOPrecovery_FSE_CMA(Input,Ni)

% Ni = Nombre d'�chantillons par symbole
%Input (1,1:15)
Nsamples = length(Input(1,:));
 power = 1/Nsamples * sum(abs(Input(1,:).^2));
 Input = 1/sqrt(power) * Input;
% Input (1,1:15)
   
%Ntaps = 10*Ni + 1;
Ntaps = 5*Ni + 1;
D=floor(Ntaps/2);                       % Retard de restitution

hxx = zeros(1,Ntaps); hxx(1,D+1) = 1.0 ; hyy = hxx;
hxy = zeros(1,Ntaps); hyx = hxy;



% M�thode 1 : Version adaptative
% ------------------------------
Naffichage = 1000;

if length(Input) > 20000
    Napprentissage = 10001;                    % Nombre d'�chantillons dans la phase apprentissage
else
    Napprentissage = floor(length(Input)/2) - Naffichage - Ni;
end;   

hxx = zeros(1,Ntaps); hxx(1,D+1) = 1.0 ; hyy = hxx;
hxy = zeros(1,Ntaps); hyx = hxy;

mu = .001; 


indice = 1;
for jter = 1:Naffichage
     Xin = Input(1,indice:(indice+Ntaps-1));
     Yin = Input(2,indice:(indice+Ntaps-1));  
     Eout(1,jter) = hxx*Xin.' + hxy*Yin.';
     Eout(2,jter) = hyx*Xin.' + hyy*Yin.';
     indice=indice+Ni;
end;  
%{
figure(51)
subplot(3,2,1)
polar(angle(Eout(1,:)), abs(Eout(1,:)),'.')
title('Constellation PolX before SOP recovery')
subplot(3,2,2)
polar(angle(Eout(2,:)), abs(Eout(2,:)),'.')
title('Constellation PolY before SOP recovery')
%}

for iter = 1:Napprentissage
       
     isample=1+(iter-1)*Ni;
     
     Xin = Input(1,isample:(isample+Ntaps-1));
     Yin = Input(2,isample:(isample+Ntaps-1));
          
     Xout = hxx*Xin.' + hxy*Yin.';
     Yout = hyx*Xin.' + hyy*Yin.';
 
     epsx = 2*(abs(Xout)^2 - 1)*Xout;
     epsy = 2*(abs(Yout)^2 - 1)*Yout;
      
     hxx = hxx - mu*epsx*conj(Xin);
     hxy = hxy - mu*epsx*conj(Yin);
     hyx = hyx - mu*epsy*conj(Xin);
     hyy = hyy - mu*epsy*conj(Yin);
    
     % affichage de l'evolution de la constellation en temps reel
     % sur 1000 Points, toutes les 500 iterations
     if (mod(iter,1000) == 1)
        ibegin = isample;
        for jter = 1:Naffichage
            Xin = Input(1,ibegin:(ibegin+Ntaps-1));
            Yin = Input(2,ibegin:(ibegin+Ntaps-1));  
            Eout(1,jter) = hxx*Xin.' + hxy*Yin.';
            Eout(2,jter) = hyx*Xin.' + hyy*Yin.';
            ibegin=ibegin+Ni;
        end;
        %{
        subplot(3,2,3)
        polar(angle(Eout(1,:)), abs(Eout(1,:)),'.')
        title('CMA + FSE')
        subplot(3,2,4)
        polar(angle(Eout(2,:)), abs(Eout(2,:)),'.')
        title(sprintf('Symbol value %3d', iter'))
        subplot(3,2,5)
        stem(abs(hxx),'r')
        hold on
        stem(-abs(hxy),'g')
        xlabel('Taps number')
        %legend('hxx','hxy')
        title('hxx & hxy')
        axis([1 Ntaps -1.2 1.2])
        hold off
        subplot(3,2,6)
        stem(abs(hyy),'r')
        hold on
        stem(-abs(hyx),'g')
        xlabel('Taps number')
        %legend('hyy','hyx')
        title('hyy & hyx')
        axis([1 Ntaps -1.2 1.2])
        hold off 
        pause(.5)
        %}

        
     end;
    
end; 

disp(sprintf('Nombre total de symboles trait�s = %d', Napprentissage))



% Etape 2 : S�quence compl�te avec egalisation calcul�e pr�c�demment
Input2 = [Input Input(:,1:Ntaps)];

Nsymbols = floor(Nsamples/Ni) - Ntaps;
Sout = zeros(2, Nsamples/Ni);
%iterIN = 1;

for iterOUT = 1:Nsymbols
     isample=1+(iterOUT-1)*Ni;
     Xin = Input(1,isample:(isample+Ntaps-1));
     Yin = Input(2,isample:(isample+Ntaps-1));   
     Sout(1,iterOUT) = hxx*Xin.' + hxy*Yin.';
     Sout(2,iterOUT) = hyx*Xin.' + hyy*Yin.'; 
end;  

%{
figure(52)
subplot(2,2,1)
polar(angle(Input(1,:)), abs(Input(1,:)),'.')
title('Constellation PolX before SOP recovery')
subplot(2,2,2)
polar(angle(Input(2,:)), abs(Input(2,:)),'.')
title('Constellation POlY before SOP recovery')
subplot(2,2,3)
polar(angle(Sout(1,:).'), abs(Sout(1,:).'),'.')
title('Constellation PolX before SOP recovery')
subplot(2,2,4)
polar(angle(Sout(2,:).'), abs(Sout(2,:).'),'.')
title('Constellation PolY before SOP recovery')
%}
disp(sprintf('Number of Input samples %g', length(Input)))
disp(sprintf('Number of Output samples %g', length(Sout)))

%Eout = sqrt(power) * Eout;
outpower_scale = sqrt(mean(abs(Sout(1, :)).^2));
Sout = Sout./outpower_scale;
Sout = circshift(Sout, [0 floor(Ntaps/2)-2]);


f = Sout;

