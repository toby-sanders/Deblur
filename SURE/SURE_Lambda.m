function [out] = SURE_Lambda(Ihat,hhat,opts)

% an automated approach to parametric deconvolution using the SURE
% criterion.  For different regularization parameters, lambda, the optimal 
% lambda is determined for simple Weiner filtered solutions.  
% The solutions are not necessarily "optimal," but this approach is used 
% because it is very fast and only used to determine deblurring parameters

% this version provides estimators of the forward error:
%%%%%    || h*(u - u_T) ||^2

% inputs: 
%   Ihat - Fourier transform of the image to deblur
%   hhat - Fourier transform of the estimated PSF
%   opts - options, including test reg. parameters, reg. operator (in 
%   Fourier domain, squared), noise level

% written by Toby Sanders @Lickenbrock Tech.
% updated: 10-20-21


[p,q] = size(Ihat);

% check input options
if ~isfield(opts,'lambdas')
    lambdas = linspace(4,0,10); 
    lambdas = 10.^(-lambdas);
else, lambdas = opts.lambdas;
end
if ~isfield(opts,'V')
    V = my_Fourier_filters(2,1,p,q,1);
else, V = opts.V;
end

% very important to know the noise level for SURE!!!
if ~isfield(opts,'sigma')
    sigma = determineNoise1D(I,10); % noise estimation
else
    sigma = opts.sigma;
end

% initialize...
nL = numel(lambdas);
out.SURE = zeros(nL,1);
sBest = 1e20;
if isfield(opts,'trueU') % check if true solution is provided for exact error
    opts.trueU = fft2(opts.trueU);
    out.SURET = zeros(nL,1);
end



hhat2 = hhat.*conj(hhat);
for j = 1:nL  % loop over lambdas (regularization parameter)
    lambda = lambdas(j);      
    % only need fourier transform of reconstruction.  The whole problem
    % stays in Fourier domain... only initial fft's are needed
    rec = Ihat.*conj(hhat)./(hhat2 + V*lambda); 
    Aub = col(rec.*hhat-Ihat);
    out.SURE(j) = Aub'*Aub/p/q + 2*sigma^2*sum(col(hhat2./(hhat2 + lambda*V)));

    % output stuff, check if true solution is provided for exact error
    if isfield(opts,'trueU')
        % out.SURET(j) = norm(col(opts.trueU) - col(ifft2(reshape(uhat,p,q))))^2;
        out.SURET(j) = sum(abs(col(hhat.*(opts.trueU - rec))).^2)/p/q;
    end

    if out.SURE(j)<sBest  % check for the minimal SURE
        sBest = out.SURE(j);
        out.rBest = rec;
        out.lambdaBest = lambda;
    end
end
% end
out.rBest = (ifft2(out.rBest)); % move solution to real domain
out.SURE = out.SURE - p*q*sigma^2; % add constant to estimator
out.sigma = sigma;
out.SUREbest = min(out.SURE);




