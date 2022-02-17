function out = SURE_FP_PSF3D(Ihat,omega,opts)

% function for finding optimal parameters for SURE deconvolution, i.e.
% we find the optimal lambda and h by the predictive estimate to the problem
%  min_u  ||h*u-b||^2 + lambda*||Du||^2

% this version finds all three parameters for the Gaussian PSF, variances
% in X and Y and an angle parameter, theta

% Written by Toby Sanders @Lickenbrock Tech.
% 10-27-21
% updated 2-17-22

[p,q] = size(Ihat);
m = p*q;

% check and set basic options
if isfield(opts, 'iter')
    iter = opts.iter;
else
    iter = 150; 
end

    
% very important to know the noise level for SURE!!!
if ~isfield(opts,'sigma')
    sigma = determineNoise1D(real(ifft2(Ihat)),10); % noise estimation
else
    sigma = opts.sigma;
end

if isfield(opts,'lambda')
    lambda = opts.lambda;
else
    lambda = 8.4e2*sigma^2/mean(col(real(ifft2(Ihat))));
    % lambda = 1/2;
end
if ~isfield(opts,'tol'), opts.tol = 0.005; end
if ~isfield(opts,'V') % power spectrum of the regularizer
    V = my_Fourier_filters(2,1,p,q,1);
else, V = opts.V;
end
if ~isfield(opts,'PSF')
    opts.PSF = 'gaus';
end

if isfield(opts,'Utrue')
    FUtrue = fft2(opts.Utrue);
end

omegaX = omega*1.0;
omegaY = omega*1.0;
theta =  0;
% initialize output variables
out.lambdas = lambda; 
out.omegasX = omegaX;
out.omegasY = omegaY;
out.thetas = theta*180/pi;
out.sigmas = [];
out.UPREs = []; % store predictive estimator
out.rel_chg = [];
out.eetrue = [];

% initialize PSF
if strcmpi(opts.PSF,'gaus')
    [~,hhat,Dx,Dy,Dt] = makeGausPSFAndDer([p,q],omegaX,omegaY,theta*180/pi,1);
    gamMax = .25;
    gamStep = .05;
elseif strcmpi(opts.PSF,'Laplace')
    [~,hhat,Dx,Dy,Dt] = makeLaplacePSFAndDer([p,q],omegaX,omegaY,theta*180/pi);
    gamMax = 2;
    gamStep = .25;
end
if isfield(opts,'gamma')
    gamMax = opts.gamma;
end
if isfield(opts,'tauIter')
    tauIter = opts.tauIter;
else
    tauIter = 3;
end
hhat2 = abs(hhat).^2;
gam = .05; % correction factor to improve fixed-point convergence
tau = 1e-2;
parmAll = [omegaX,omegaY,lambda,theta];
for i = 1:iter
    
    % compute ratio needed for omega update
    % numerator is the trace term and denomenator is the long matrix vector
    % inner product
    denom = 1 - hhat2./(hhat2 + lambda*V);
    denom0 = denom.*Ihat;
    denom = denom0.*hhat./(hhat2 + lambda*V);
    denomX = denom.*Dx;
    denomY = denom.*Dy;
    denomT = denom.*Dt;
    denomX = -sum(conj(denom0(:)).*denomX(:))/m;
    denomY = -sum(conj(denom0(:)).*denomY(:))/m;
    denomT = -sum(conj(denom0(:)).*denomT(:))/m;
    numer = sigma^2*lambda*V.*hhat./(hhat2 + lambda*V).^2;
    numerX = sum(Dx(:).*numer(:));
    numerY = sum(Dy(:).*numer(:));
    numerT = sum(Dt(:).*numer(:));
    omegaX = omegaX*real(-numerX/denomX)^gam;
    omegaY = omegaY*real(-numerY/denomY)^gam;

    if i>tauIter
        Snum = theta - thetap;
        Yden = real(numerT + denomT) - gT;
        tau = Snum/Yden;
    end
    thetap = theta;
    gT = real(numerT + denomT);
    theta =  theta - tau*gT;% 
    % theta = theta*real(-numerT/denomT)^gam2;
    theta = theta - round(theta/(pi/2))*pi/2;
    % update PSF
    if strcmpi(opts.PSF,'gaus')
        [~,hhat,Dx,Dy,Dt] = makeGausPSFAndDer([p,q],omegaX,omegaY,theta*180/pi,1);
    elseif strcmpi(opts.PSF,'Laplace')
        [~,hhat,Dx,Dy,Dt] = makeLaplacePSFAndDer([p,q],omegaX,omegaY,theta*180/pi);
    end
    hhat2 = abs(hhat).^2;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update lambda below
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Wiener solution and trace of H^(-2) D^T D A^T A
    rec = Ihat.*conj(hhat)./(hhat2+ V*lambda);
    traceLambda = sum(hhat2(:).*V(:)./(hhat2(:)+V(:)*lambda).^2);

    % new lambda updated based on the fixed point
    denom = V.*rec; denom2 = denom./(hhat2+V*lambda);
    lambda = real((m*sigma^2*traceLambda)/sum(col(conj(denom).*denom2)));
    
    % update sigma if unknown and only if m>trapp
    % if Fsig && m>trHiAA, sigma = sqrt(re'*re/(m-trHiAA)); end

    % upre stuff which is NOT necessary, this is for output only
    rec = Ihat.*conj(hhat)./(hhat2+ V*lambda);
    Aub = col(rec.*hhat - Ihat);
    trHiAA = sum(hhat2(:)./(V(:)*lambda + hhat2(:)));
    out.UPREs = [out.UPREs;-m*sigma^2 + (Aub'*Aub)/m + 2*sigma^2*trHiAA];
    % compute SURE and outputs
    if isfield(opts,'Utrue')
        out.eetrue =  [out.eetrue;1/m*norm(col((hhat.*(rec - FUtrue)))).^2];
        % [out.eetrue;1/m*norm(col((opts.hhatT.*(rec - FUtrue)))).^2];
    end    
    out.lambdas = [out.lambdas;lambda];
    out.sigmas = [out.sigmas;sigma];
    out.omegasX = [out.omegasX;omegaX];
    out.omegasY = [out.omegasY;omegaY];
    out.thetas = [out.thetas;theta*180/pi];

    parmAllp = parmAll;
    parmAll = [omegaX,omegaY,lambda,theta];
    % check for convergence
    
    rel_chg = abs(parmAllp-parmAll);
    out.rel_chg = [out.rel_chg;rel_chg];
    if rel_chg<opts.tol, break; end



    gam = min(gam+(gamStep),gamMax);
    
end
out.U = real(ifftn(rec));
out.hhat = hhat;
% out.thetas = out.thetas - round(out.thetas/90)*90;

