function [psf,Fpsf,DhX,DhY,DhT] = makeGausPSF(n,sigmaX,sigmaY,theta,pixelSize)

% make a two dimension Gaussian distribution with n pixels and a variance
% of sigma^2, where each pixel is considered 1 unit length

if numel(n)==1, d1 = n; d2 = n;
elseif numel(n)==2, d1 = n(1); d2 = n(2);
else, error('too many dimensions in n');
end
if nargin<4
    theta = 0;
    pixelSize = 1;
elseif nargin<5
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
[x,y] = meshgrid(x,y);
X = cosd(theta)*x + sind(theta)*y;
Y = -sind(theta)*x + cosd(theta)*y;
XiX = cosd(theta)*x/d2 + sind(theta)*y/d1;
XiY = -sind(theta)*x/d2 + cosd(theta)*y/d1;

psf = exp(-X.^2/(2*sigmaX^2) - Y.^2/(2*sigmaY^2));
psf = ifftshift(psf/sum(sum(psf)));

% Fourier transform of PSF, also Gaussian, derived analytically
Fpsf = ifftshift(exp((-2*sigmaX^2*(pi*XiX).^2  - 2*sigmaY^2*(pi*XiY).^2)/pixelSize^4));


% derivative of Fpsf, with respect to sigmaX, sigmaY, and theta
Xi = x/d2;
Yi = y/d1;
Dh = fftshift(Fpsf);
DhX = ifftshift(-4*sigmaX*pi^2*(cosd(theta)*Xi + sind(theta)*Yi).^2.*Dh);
DhY = ifftshift(-4*sigmaY*pi^2*(-sind(theta)*Xi + cosd(theta)*Yi).^2.*Dh);
DhT = ifftshift(-4*pi^2*(sigmaX^2 - sigmaY^2)*(cosd(theta)*Xi + sind(theta)*Yi).*(-sind(theta)*Xi + cosd(theta)*Yi).*Dh);




