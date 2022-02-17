function [psf,Fpsf] = makeGausPSF2D(n,sigmaX,sigmaY)

% make a two dimension Gaussian distribution with n pixels and a variance
% of sigmaX and sigmaY, where each pixel is considered 1 unit length

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

psf = exp(-X.^2/(2*sigmaX^2) - Y.^2/(2*sigmaY^2));
psf = ifftshift(psf/sum(sum(psf)));

% Fourier transform of PSF, also Gaussian, derived analytically
Fpsf = ifftshift(exp(-2*sigmaX^2*(pi*X/d2).^2 - 2*sigmaY^2*(pi*Y/d1).^2));