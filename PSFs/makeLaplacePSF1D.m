function [psf,Fpsf] = makeLaplacePSF1D(n,sigma)

% make a two dimension Gaussian distribution with n pixels and a variance
% of sigmaX and sigmaY, where each pixel is considered 1 unit length

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

psf = exp(-abs(x)*sqrt(2)/sigma);
% exp(-X.^2/(2*sigmaX^2) - Y.^2/(2*sigmaY^2));
psf = ifftshift(psf/sum(psf(:)));

% Fourier transform of PSF
Fpsf = fft2(psf);