function out = SURE_Pois_FP_PSF_multiPSF(Ihat,fhat,opts)


% function for finding optimal parameters for SURE deconvolution, i.e.
% we find the optimal lambda and h by the predictive estimate to the problem
%  min_u  ||h*u-b||^2 + lambda*||Du||^2

% this version finds the coefficients for linearly combining a large number
% of PSFs, which are stacked into fhat

% Written by Toby Sanders @Lickenbrock Tech.
% 1-28-2022

[p,q,N] = size(fhat);
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
% if ~isfield(opts,'sigma')
%     sigma = determineNoise1D(I,10); % noise estimation
% else
%     sigma = opts.sigma;
% end
if ~isfield(opts,'tol'), opts.tol = 0.005; end
if ~isfield(opts,'V') % power spectrum of the regularizer
    V = my_Fourier_filters(2,1,p,q,1);
else, V = opts.V;
end

alpha = 1/N*ones(N,1);
% alpha0 = .05;
% alpha(1) = alpha0;
% alpha(2:end) = (1-alpha0)/(N-1);

% initialize output variables
out.lambdas = lambda; 
out.alphas = alpha;
% out.sigmas = [];
out.UPREs = []; % store predictive estimator
out.rel_chg = [];
out.eetrue = [];
out.normer = [];
out.numUsed = N;

% initialize PSF
hhat = zeros(p,q);
for j = 1:N
    hhat = hhat + alpha(j)*fhat(:,:,j);
end

hhat2 = abs(hhat).^2;
gam = 0.5; % correction factor to improve fixed-point convergence
alphaTolerance = 1e-1/N;


SS = 1:N;
for i = 1:iter
    
    % precompute values needed for all alpha coefficients
    M = 1./(hhat2 + lambda*V);
    denom0 = abs((1 - hhat2.*M).*Ihat).^2;
    numer = (Ihat(1)/m)*lambda*V.*hhat.*(M.^2);

    % update alpha coefficients
    for j = SS%  1:N
        tmp = real(hhat.*conj(fhat(:,:,j)).*M);
        denomF = -sum(tmp(:).*denom0(:))/m;
        numerF = sum(real(col(fhat(:,:,j).*numer)));
        alpha(j) = alpha(j).*real(-denomF/numerF)^gam;
    end
    
    % throw out very small coefficients to reduce computations
    SS = find(alpha>alphaTolerance)';
    SS2 = alpha<=alphaTolerance;
    alpha(SS2) = 0;
    
    % normalize the coefficients
    normer = sum(alpha);
    alpha = alpha/sum(alpha);

    % update PSF
    hhat(:) = 0;
    for j = SS% 1:N
        hhat = hhat + alpha(j)*fhat(:,:,j);
    end
    hhat2 = abs(hhat).^2;
   
    % Wiener solution and error
    numer = Ihat.*conj(hhat);
    denom = hhat2 + V*lambda;
    % rec = Ihat.*conj(hhat)./(hhat2+ V*lambda);

    % trace of H^(-2) D^T D A^T A
    traceLambda = Ihat(1)/m*sum(hhat2(:).*V(:)./denom(:).^2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update lambda below
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % new lambda updated based on the fixed point
    denom1 = V.*numer./denom;% rec; 
    denom2 = denom1./denom;% (hhat2+V*lambda);
    lambda = real((m*traceLambda)/sum(col(conj(denom1).*denom2)));

    % Wiener solution and error
    denom = hhat2 + V*lambda;
    rec = numer./denom;
    Aub = col(rec.*hhat - Ihat);

    % update sigma if unknown and only if m>trapp
    % if Fsig && m>trHiAA, sigma = sqrt(re'*re/(m-trHiAA)); end
    % trHiAA = sum(hhat2(:)./(V(:)*lambda + hhat2(:)));
    trHiAA = sum(hhat2(:)./denom(:));
    out.UPREs = [out.UPREs;-Ihat(1) + (Aub'*Aub)/m + 2*Ihat(1)/m*trHiAA];
    % compute SURE and outputs
    if isfield(opts,'Utrue')    
        out.eetrue = [out.eetrue;norm(col(ifftn(Khat.*fftn(ifftn(Fu)-opts.Utrue))))^2];
    end    
    out.lambdas = [out.lambdas;lambda];
    % out.sigmas = [out.sigmas;sigma];
    out.alphas = [out.alphas,alpha];
    out.numUsed = [out.numUsed;numel(SS)];
    % out.betas = [out.betas;beta];
    out.normer = [out.normer;normer];
    
    % check for convergence
    rel_chg = sum(abs(alpha-out.alphas(:,end-1)))/N;
    out.rel_chg = [out.rel_chg;rel_chg];
    if rel_chg<opts.tol, break; end
end
out.U = real(ifftn(rec));
out.hhat = hhat;
