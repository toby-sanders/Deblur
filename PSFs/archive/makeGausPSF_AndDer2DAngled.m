function [Fpsf,DhX,DhY,DhT] = makeGausPSF(n,sigmaX,sigmaY,theta)

% make a two dimension Gaussian distribution with n pixels and a variance
% of sigma^2, where each pixel is considered 1 unit length

if numel(n)==1, d1 = n; d2 = n;
elseif numel(n)==2, d1 = n(1); d2 = n(2);
else, error('too many dimensions in n');
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

X2 = cosd(theta)*X + sind(theta)*Y;
Y2 = -sind(theta)*X + cosd(theta)*Y;
psf = exp(-X2.^2/(2*sigmaX^2) - Y2.^2/(2*sigmaY^2));
psf = ifftshift(psf/sum(sum(psf)));

% Fourier transform of PSF, also Gaussian, derived analytically
Fpsf = ifftshift(exp(-2*sigmaX^2*(pi*X2/d2).^2 - 2*sigmaY^2*(pi*Y2/d1).^2));

% derivative of Fpsf, with respect to sigmaX
Dh = fftshift(Fpsf);
DhX = ifftshift(-4*sigmaX*pi^2*(cosd(theta)*Xi + sind(theta)*Yi).^2.*Dh);
DhY = ifftshift(-4*sigmaY*pi^2*(-sind(theta)*Xi + cosd(theta)*Yi).^2.*Dh);
DhT = ifftshift(-4*pi^2*(sigmaX^2 - sigmaY^2)*(cosd(theta)*Xi + sind(theta)*Yi).*(-sind(theta)*Xi + cosd(theta)*Yi).*Dh);