function samples_out = RC_RRC_for_FTN(beta,nr_samples,tau,rc_rrc_string,SPS_factor_RRC)
% t = n*SPS_factor_RRC*tau*T_s
% => SPS_factor_RRC <= 1/tau/(1+beta) for proper sampling of RRC function; SPS_factor_RRC = 0.5 works for all combinations of beta and tau
tau_vector = tau*(-(nr_samples-1)/2:(nr_samples-1)/2);
switch rc_rrc_string
    case 'normal'
        
        % Added later +++
        if SPS_factor_RRC < 1
            nr_samples = (nr_samples-1)/SPS_factor_RRC+1;
            tau_vector = tau*(-(nr_samples-1)/2:(nr_samples-1)/2)*SPS_factor_RRC;
        end
        % Added later+++
        
        if ~mod(nr_samples,2)
            error('Number of samples must be odd');
        end     
        
        samples_out = sinc(tau_vector).*cos(pi*beta*tau_vector)./(1-4*beta^2*tau_vector.^2);
    case 'sqrt'
        if SPS_factor_RRC < 1
            nr_samples = (nr_samples-1)/SPS_factor_RRC+1;
            tau_vector = tau*(-(nr_samples-1)/2:(nr_samples-1)/2)*SPS_factor_RRC;
        end
        if ~mod(nr_samples,2)
            error('Number of samples must be odd');
        end
        samples_out = (sin(pi*tau_vector*(1-beta))+4*beta*tau_vector.*cos(pi*tau_vector*(1+beta)))./(pi*tau_vector.*(1-16*beta^2*tau_vector.^2));
        zero_idx = find(tau_vector==0);
        samples_out(zero_idx) = 1-beta+4*beta/pi;
        other_indices = find((tau_vector == 1/4/beta)|(tau_vector == -1/4/beta));
        samples_out(other_indices) = beta/sqrt(2)*((1+2/pi)*sin(pi/4/beta)+(1-2/pi)*cos(pi/4/beta));
        samples_out = samples_out/norm(samples_out);
    otherwise
        error('only ''normal'' and ''sqrt'' strings work')
end
return