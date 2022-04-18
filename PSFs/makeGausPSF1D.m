function [psf,Fpsf] = makeGausPSF1D(n,sigma)

% make a one dimension Gaussian distribution with n pixels and a variance
% of sigma^2, where each pixel is considered 1 unit length

if numel(n)~=1
    error('too many dimensions in n');
end

% make centered meshgrid
if mod(n,2)==1
    xBound = pixelSize*(n-1)/2;
    x = linspace(-xBound,xBound,n)';
else
    xBound = pixelSize*n/2;
    x = linspace(-xBound,xBound-pixelSize,n)';
end
psf = exp(-x.^2/(2*sigma^2));
psf = ifftshift(psf/sum(sum(psf)));

% Fourier transform of PSF, also Gaussian, derived analytically
Fpsf = ifftshift(exp(-2*sigma^2*(pi*x/n).^2));