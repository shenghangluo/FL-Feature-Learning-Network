function f = xQAMsignal(Sequence_baud, Nsamples_baud)

signal = zeros(1, Nsamples_baud*length(Sequence_baud)); 
%Sequence_baud = Sequence_baud*exp(1i*Offset_baud);
for iter = 1:length(Sequence_baud)
    signal([((iter-1)*Nsamples_baud+1):(iter*Nsamples_baud)]) = Sequence_baud(iter);
end; 
f = signal;