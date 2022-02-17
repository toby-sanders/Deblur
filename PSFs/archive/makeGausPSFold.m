function [psf,Fpsf] = makeGausPSF(n,sigma,pixelSize)

% make a two dimension Gaussian distribution with n pixels and a variance
% of sigma^2, where each pixel is considered 1 unit length

if numel(n)==1, d1 = n; d2 = n;
elseif numel(n)==2, d1 = n(1); d2 = n(2);
else, error('too many dimensions in n');
end

if nargin<3
    pixelSize = 1;
end

xBound = pixelSize*d2/2;
yBound = pixelSize*d1/2;

% make centered meshgrid
if mod(d2,2)==1, x = linspace(-xBound,xBound,d2)';
else, x = linspace(-xBound,xBound-pixelSize,d2)';
end
if mod(d1,2)==1, y = linspace(-yBound,yBound,d1)';
else, y = linspace(-yBound,yBound-pixelSize,d1)';
end
[X,Y] = meshgrid(x,y);

psf = exp(-(X.^2+Y.^2)/(2*sigma^2));
psf = ifftshift(psf/sum(sum(psf)));

% Fourier transform of PSF, also Gaussian, derived analytically
Fpsf = ifftshift(exp(-2*sigma^2*((pi*X/d2).^2 + (pi*Y/d1).^2)/pixelSize^4));