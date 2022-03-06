function [psf,psf1,psf2] = makeLangPSF(n,pixelSize)

% make a 1D Langevian function

if numel(n)==1, d1 = n; d2 = n;
elseif numel(n)==2, d1 = n(1); d2 = n(2);
else, error('too many dimensions in n');
end
if nargin<2
    pixelSize = .1;
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

psf1 = coth(x) - 1./x;
psf2 = x/3 - x.^3/45 + 2*x.^5/945 - x.^7/4725;

S = abs(x)<1/2;
psf = psf1;
psf(S) = psf2(S);