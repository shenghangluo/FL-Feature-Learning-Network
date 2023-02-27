function [Out_Ax, Out_Ay] = CD_Compensation_Lumped2(Ax, Ay, cf, Be, L, Chrom_Disp, lambda0)

   
    % Constants/Constantes
    i=sqrt(-1);
    c = 3e8;
 
    beta2 = -(lambda0*1e-9)^2/(2*pi*c) * Chrom_Disp*1e-6;         % s^2/m

    % definition of the frequency axis/definition de l'axe des frequences
    totalsamples = length(Ax);
    f = (-totalsamples/2+1:totalsamples/2)/totalsamples;
    f = cf+fftshift(f) * Be; 
    w = 2*pi.*f; 

    AX = fft(Ax); AY = fft(Ay);

    % dispersion
    Disp = exp(i/2 * beta2 .* w.^2 .* L); 
    AX = AX .* Disp; AY = AY .* Disp;  

    Ax_inter = ifft(AX); Ay_inter = ifft(AY);

    Out_Ax = Ax_inter;
    Out_Ay = Ay_inter;
end