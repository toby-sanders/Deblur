function [U,out] = SURE_FP_Lambda(Ihat,hhat,opts)


% function for finding optimal parameter from UPRE
% i.e. We find the optimal mu by the predictive estimate to the problem
%  min_u  mu||Au-b||^2 + ||Du||^2
% fixed point version

% Written by Toby Sanders @ASU
% School of Math & Stat Sciences
% August 2018
% Updated: 02-17-22


if isfield(opts, 'iter'); iter = opts.iter;
else, iter = 50; end
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
[p,q,r] = size(Ihat);
m = p*q*r;
if ~isfield(opts,'tol'), opts.tol = 1e-3; end
% [D,Dt] = get_D_Dt(opts.order,p,q,r,opts,b);

if ~isfield(opts,'V')
    V = my_Fourier_filters(2,1,p,q,1);
else, V = opts.V;
end

hhat2 = abs(hhat).^2;
hCI = conj(hhat).*Ihat;

out.lambdas = lambda; 
out.sigmas = [];
out.UPREs = []; % store predictive estimator
out.rel_chg = [];
% if Utrue supplied can compute exact predictive error
if isfield(opts,'Utrue'), out.eetrue = []; end
for i = 1:iter
    % solve Tikhonov reg. problem for each lambda
    rec = hCI./(hhat2+ V*lambda);
    
    % u = real(ifftn(Fu)); % actually do not need u
    Aub = col(rec.*hhat - Ihat);
    
    % trace of  H^-1 A^T A
    trHiAA = sum(hhat2(:)./(V(:)*lambda + hhat2(:)));

    % trace of H^(-2) D^T D A^T A
    trBIG = sum(hhat2(:).*V(:)./(hhat2(:)+V(:)*lambda).^2);
    
    % update sigma if unknown and only if m>trapp
    % if Fsig && m>trHiAA, sigma = sqrt(re'*re/(m-trHiAA)); end
    
    % compute UPRE
    out.UPREs = [out.UPREs;-m*sigma^2 + (Aub'*Aub)/m + 2*sigma^2*trHiAA];
    if isfield(opts,'Utrue')
        out.eetrue = [out.eetrue;norm(col(ifftn(Khat.*fftn(ifftn(Fu)-opts.Utrue))))^2];
    end    
    
    % new lambda, updated based on the fixed point
    tmp = V.*rec; tmp2 = tmp./(hhat2+V*lambda);

    % the "m" in the numerator is the normalization needed since the inner
    % product term in the denomenator is computed in non-unitary Fourier
    % domain. All of the trace terms computed in Fourier domain are fine.
    lambda = real((sigma^2*trBIG)*(m)/sum(col(conj(tmp).*tmp2)));
    out.lambdas = [out.lambdas;lambda];
    out.sigmas = [out.sigmas;sigma];
    rel_chg = abs(out.lambdas(end-1)-lambda)/lambda;
    out.rel_chg = [out.rel_chg;rel_chg];
    if rel_chg<opts.tol, break; end
end
U = real(ifftn(rec));

