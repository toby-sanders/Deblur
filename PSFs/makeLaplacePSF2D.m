function [psf,Fpsf] = makeLaplacePSF2D(n,sigmaX,sigmaY,theta)

% make a two dimension Gaussian distribution with n pixels and a variance
% of sigmaX and sigmaY, where each pixel is considered 1 unit length

if numel(n)==1, d1 = n; d2 = n;
elseif numel(n)==2, d1 = n(1); d2 = n(2);
else, error('too many dimensions in n');
end


% make centered meshgrid
if mod(d2,2)==1
    xBound = pixelSize*(d2-1)/2;
    x = linspace(-xBound,xBound,d2)';
else
    xBound = pixelSize*d2/2;
    x = linspace(-xBound,xBound-pixelSize,d2)';
end
if mod(d1,2)==1
    yBound = pixelSize*(d1-1)/2;
    y = linspace(-yBound,yBound,d1)';
else
    yBound = pixelSize*d1/2;
    y = linspace(-yBound,yBound-pixelSize,d1)';
end
[x,y] = meshgrid(x,y);
X = cosd(theta)*x + sind(theta)*y;
Y = -sind(theta)*x + cosd(theta)*y;

psf = exp(-abs(X)*sqrt(2)/sigmaX - abs(Y)*sqrt(2)/sigmaY);
psf = ifftshift(psf/sum(psf(:)));

% Fourier transform of PSF
Fpsf = fft2(psf);