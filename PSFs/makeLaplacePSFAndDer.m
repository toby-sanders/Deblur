function [psf,Fpsf,Dx,Dy,Dtheta] = makeLaplacePSF2D(n,alphaX,alphaY,theta)

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
X = cosd(theta)*x/d2 + sind(theta)*y/d1;
Y = -sind(theta)*x/d2 + cosd(theta)*y/d1;
X = ifftshift(X);
Y = ifftshift(Y);

fx = 1+alphaX*sin(pi*X).^2;
fy = 1+alphaY*sin(pi*Y).^2;
Fpsf = (fx.*fy).^(-1);
Dx = -(sin(pi*X).^2).*((fy.*fx.^2).^(-1));
Dy = -(sin(pi*Y).^2).*((fx.*fy.^2).^(-1));
Dtheta1 = -(2*pi*alphaX)*sin(pi*X).*cos(pi*X).*Y.*((fy.*fx.^2).^(-1));
Dtheta2 = (2*pi*alphaY)*sin(pi*Y).*cos(pi*Y).*X.*((fx.*fy.^2).^(-1));
Dtheta = Dtheta1 + Dtheta2;

psf = [];