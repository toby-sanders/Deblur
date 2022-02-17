function out = SURE_FP_PSF1D_only(Ihat,omega,opts)


% function for finding optimal PSF for SURE deconvolution, i.e.
% we find the optimal PSF, h, by the predictive estimate to the problem
%  min_u  ||h*u-b||^2 + lambda*||Du||^2

% this version finds one parameter for the Gaussian PSF, the variance
% this version does NOT find the optimal lambda in the Wiener filter

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
if ~isfield(opts,'tol'), opts.tol = 0.05; end
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
gam = 1/2;
for i = 1:iter
    % update PSF
    [~,hhat,Dh] = makeGausPSF_AndDer([p,q],omega,1);
    hhat2 = abs(hhat).^2;
    
    % compute ratio needed for omega update
    % numerator is the trace term and denomenator is the long matrix vector
    % inner product
    denom = 1 - hhat2./(hhat2 + lambda*V);
    denom = denom.*Ihat;
    denom2 = denom.*(hhat.*Dh)./(hhat2 + lambda*V);
    denom = -sum(conj(denom(:)).*denom2(:))/m;

    numer = sigma^2*lambda*sum(sum(V.*hhat.*Dh./(hhat2 + lambda*V).^2));

    omega = omega*real(-numer/denom)^gam;
    out.omegas = [out.omegas;omega];
end
% U = real(ifftn(rec));
out.U = real(ifft2(Ihat.*conj(hhat)./(hhat2 + lambda*V)));
