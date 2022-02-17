function [psf,Fpsf] = makeLaplacePSF2D(n,sigmaX,sigmaY,theta)

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
[x,y] = meshgrid(x,y);
X = cosd(theta)*x + sind(theta)*y;
Y = -sind(theta)*x + cosd(theta)*y;

psf = exp(-abs(X)*sqrt(2)/sigmaX - abs(Y)*sqrt(2)/sigmaY);
psf = ifftshift(psf/sum(psf(:)));

% Fourier transform of PSF
Fpsf = fft2(psf);