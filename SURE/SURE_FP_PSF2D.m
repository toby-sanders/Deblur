function out = SURE_FP_PSF2D(Ihat,omega,opts)


% function for finding optimal parameters for SURE deconvolution, i.e.
% we find the optimal lambda and h by the predictive estimate to the problem
%  min_u  ||h*u-b||^2 + lambda*||Du||^2

% this version finds two parameters for the Gaussian PSF, variances
% in X and Y 

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
out.omegasX = omega;omegaX = omega;
out.omegasY = omega;omegaY = omega;
out.sigmas = [];
out.UPREs = []; % store predictive estimator
out.rel_chg = [];
out.eetrue = [];

% initialize PSF
[~,hhat,Dx,Dy] = makeGausPSFAndDer([p,q],omegaX,omegaY,0);
hhat2 = abs(hhat).^2;
gam = 1/2; % correction factor to improve fixed-point convergence
for i = 1:iter
    % compute ratio needed for omega update
    % numerator is the trace term and denomenator is the long matrix vector
    % inner product
    denom = 1 - hhat2./(hhat2 + lambda*V);
    denom0 = denom.*Ihat;
    denom = denom0.*hhat./(hhat2 + lambda*V);
    denomX = denom.*Dx;
    denomY = denom.*Dy;
    denomX = -sum(conj(denom0(:)).*denomX(:))/m;
    denomY = -sum(conj(denom0(:)).*denomY(:))/m;
    numer = sigma^2*lambda*V.*hhat./(hhat2 + lambda*V).^2;
    numerX = sum(Dx(:).*numer(:));
    numerY = sum(Dy(:).*numer(:));
    omegaX = omegaX*real(-numerX/denomX)^gam;
    omegaY = omegaY*real(-numerY/denomY)^gam;
    
    
    % update PSF
    [~,hhat,Dx,Dy] = makeGausPSFAndDer([p,q],omegaX,omegaY,0);
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
    lambda = real((m*sigma^2*traceLambda)/sum(col(conj(denom).*denom2)));

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
    out.omegasX = [out.omegasX;omegaX];
    out.omegasY = [out.omegasY;omegaY];

    % check for convergence
    rel_chg = abs(omegaX - out.omegasX(end-1));
    out.rel_chg = [out.rel_chg;rel_chg];
    if rel_chg<opts.tol, break; end
end
out.U = real(ifftn(rec));

