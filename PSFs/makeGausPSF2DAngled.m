function [psf,Fpsf] = makeGausPSF2D(n,sigmaX,sigmaY,theta)

% make a two dimension Gaussian distribution with n pixels and a variance
% of sigmaX and sigmaY, where each pixel is considered 1 unit length

if numel(n)==1, d1 = n; d2 = n;
elseif numel(n)==2, d1 = n(1); d2 = n(2);
else, error('too many dimensions in n');
end
if nargin<4
    theta = 0;
end

% make centered meshgrid
if mod(d2,2)==1, x = linspace(-d2/2,d2/2,d2)';
else, x = linspace(-d2/2,d2/2-1,d2)';
end
if mod(d1,2)==1, y = linspace(-d1/2,d1/2,d1)';
else, y = linspace(-d1/2,d1/2-1,d1)';
end
[X,Y] = meshgrid(x,y);
Xi = X/d2;
Yi = Y/d1;

method = 2;

% two methods for generating the angled point spread function
% method 1 uses the long derived expression for the PSF in terms of X and Y
% and their coefficents
% method 2 simply rotates the X-Y coordinate system and uses the
% traditional formulas
if method == 1
    a = (cosd(theta)/sigmaX)^2 + (sind(theta)/sigmaY)^2;
    b = (sind(theta)/sigmaX)^2 + (cosd(theta)/sigmaY)^2;
    c = cosd(theta)*sind(theta)*(sigmaX^(-2) - sigmaY^(-2));
    psf = exp(-(a/2)*X.^2 - (b/2)*Y.^2 - c*X.*Y);
    psf = ifftshift(psf/sum(sum(psf)));
    
    % Fourier transform of PSF, also Gaussian, derived analytically
    alpha = 2*pi^2*((sigmaX*cosd(theta))^2 + (sigmaY*sind(theta))^2);
    beta = 2*pi^2*((sigmaX*sind(theta))^2 + (sigmaY*cosd(theta))^2);
    gam = 4*pi^2*cosd(theta)*sind(theta)*(sigmaX^2 - sigmaY^2);
    Fpsf = ifftshift(exp(-alpha*Xi.^2 - beta*Yi.^2 - gam*Xi.*Yi));

elseif method == 2
    X2 = cosd(theta)*X + sind(theta)*Y;
    Y2 = -sind(theta)*X + cosd(theta)*Y;
    psf = exp(-X2.^2/(2*sigmaX^2) - Y2.^2/(2*sigmaY^2));
    psf = ifftshift(psf/sum(sum(psf)));
    
    % Fourier transform of PSF, also Gaussian, derived analytically
    Fpsf = ifftshift(exp(-2*sigmaX^2*(pi*X2/d2).^2 - 2*sigmaY^2*(pi*Y2/d1).^2));
end

