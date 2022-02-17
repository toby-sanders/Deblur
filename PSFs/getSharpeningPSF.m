function [h,hhat] = getSharpeningPSF(m,n,alpha,lambda)

if nargin<4
    lambda = 1e-5;
end

T = [0 -1 0; -1 4 -1; 0 -1 0];
% T = [-1 -1 -1; -1 8 -1; -1 -1 -1];
T2 = zeros(m,n);
T2(1:3,1:3) = T;
Lfilt = 1 + alpha*fft2(circshift(T2,[-1 -1]),m,n);

a = Lfilt;
b = -1;
c = lambda*Lfilt;

testConstraint = max(4.*a(:).*c(:));
if testConstraint>1
    error('constraint not met');
end

hhat = (-b+sqrt(b^2 - 4.*a.*c))./(2*a);
% hhat2 = (-b-sqrt(b^2 - 4.*a.*c))./(2*a);
h = ifft2(hhat);




