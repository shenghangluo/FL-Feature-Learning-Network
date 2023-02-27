function my_plot_fft(Time_domain_sym, Total_upsamling_factor, BaudRate, str)

    Xmsg = fftshift(fft(Time_domain_sym));
    totalsamples = length(Xmsg);
    f = Total_upsamling_factor*BaudRate*(-totalsamples/2+1:totalsamples/2)/totalsamples;
    XmsgMaxabs = max(abs(Xmsg));
    %figure
    switch (str)
        case 'norm'
            plot(f,20*log10(abs(Xmsg)./XmsgMaxabs));
            xlabel('Baseband Frequency');
            ylabel('Normalized Power in dB');
        case 'log'
            semilogy(f,20*log10(abs(Xmsg)));
             xlabel('Baseband Frequency');
             ylabel('Power in dB');
    end      
    
    
end
    