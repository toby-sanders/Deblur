function [psf] = makeGausPSF1D(n,ppp)

% make a one dimension Gaussian distribution with n pixels and a variance
% of sigma^2, where each pixel is considered 1 unit length

if numel(n)~=1
    error('too many dimensions in n');
end

% make centered meshgrid
if mod(n,2)==1
    xBound = (n-1)/2;
    x = linspace(-xBound,xBound,n)';
else
    xBound = n/2;
    x = linspace(-xBound,xBound-1,n)';
end
S = abs(x)<round(xBound*(1-ppp));
psf = double(S);
