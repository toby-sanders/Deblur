function out = SURE_FP_PSF1D(Ihat,omega,opts)


% function for finding optimal parameters for SURE deconvolution, i.e.
% we find the optimal lambda and h by the predictive estimate to the problem
%  min_u  ||h*u-b||^2 + lambda*||Du||^2

% this version finds one parameter for the Gaussian PSF, the variance
% it also finds the optimal lambda in the Wiener filter

% Written by Toby Sanders @Lickenbrock Tech.
% 10-27-21

[p,q] = size(Ihat);
m = p*q;

% check and set basic options
if isfield(opts, 'iter')
    iter = opts.iter;
else
    iter = 150; 
end
if isfield(opts,'lambda')
    lambda = opts.lambda;
else, lambda = 0.5;
end
    
% very important to know the noise level for SURE!!!
if ~isfield(opts,'sigma')
    sigma = determineNoise1D(I,10); % noise estimation
else
    sigma = opts.sigma;
end
if ~isfield(opts,'tol'), opts.tol = 0.005; end
if ~isfield(opts,'V') % power spectrum of the regularizer
    V = my_Fourier_filters(2,1,p,q,1);
else, V = opts.V;
end

% initialize output variables
out.lambdas = lambda; 
out.omegas = omega;
out.sigmas = [];
out.UPREs = []; % store predictive estimator
out.rel_chg = [];
out.eetrue = [];

% initialize PSF
[~,hhat,Dh] = makeGausPSF_AndDer([p,q],omega,1);
hhat2 = abs(hhat).^2;
gam = 1/2; % correction factor to improve fixed-point convergence
for i = 1:iter
    % compute ratio needed for omega update
    % numerator is the trace term and denomenator is the long matrix vector
    % inner product
    denom = 1 - hhat2./(hhat2 + lambda*V);
    denom = denom.*Ihat;
    denom2 = denom.*(hhat.*Dh)./(hhat2 + lambda*V);
    denom = -sum(conj(denom(:)).*denom2(:))/m;
    numer = sigma^2*lambda*sum(sum(V.*hhat.*Dh./(hhat2 + lambda*V).^2));
    omega = omega*real(-numer/denom)^gam;
    
    % update PSF
    [~,hhat,Dh] = makeGausPSF_AndDer([p,q],omega,1);
    hhat2 = abs(hhat).^2;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update lambda below
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Wiener solution and error
    rec = Ihat.*conj(hhat)./(hhat2+ V*lambda);
    Aub = col(rec.*hhat - Ihat);

    % trace of H^(-2) D^T D A^T A
    traceLambda = sum(hhat2(:).*V(:)./(hhat2(:)+V(:)*lambda).^2);

    % new lambda updated based on the fixed point
    denom = V.*rec; denom2 = denom./(hhat2+V*lambda);
    lambda = (m*sigma^2*traceLambda)/sum(col(conj(denom).*denom2));

    % update sigma if unknown and only if m>trapp
    % if Fsig && m>trHiAA, sigma = sqrt(re'*re/(m-trHiAA)); end
    
    % compute SURE and outputs
    if isfield(opts,'Utrue')
        trHiAA = sum(hhat2(:)./(V(:)*lambda + hhat2(:)));
        out.UPREs = [out.UPREs;-m*sigma^2 + (Aub'*Aub)/m + 2*sigma^2*trHiAA];
        out.eetrue = [out.eetrue;norm(col(ifftn(Khat.*fftn(ifftn(Fu)-opts.Utrue))))^2];
    end    
    out.lambdas = [out.lambdas;lambda];
    out.sigmas = [out.sigmas;sigma];
    out.omegas = [out.omegas;omega];

    % check for convergence
    rel_chg = abs(omega - out.omegas(end-1));
    out.rel_chg = [out.rel_chg;rel_chg];
    if rel_chg<opts.tol, break; end
end
out.U = real(ifftn(rec));

