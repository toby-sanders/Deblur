function y = SUREobj(params,d1,d2,bhat,V,sigma)

% compute the value of SURE for the given input parameters
% params is a 4x1 array with the values (omegaX,omegaY,theta,lambda)

[~,hhat] = makeGausPSF([d1,d2],params(1),params(2),params(3)*180/pi,1);
lambda = params(4);
hhat2 = abs(hhat).^2;
M = 1./(hhat2 + lambda*V);

rec = bhat.*conj(hhat).*M;
Aub = col(rec.*hhat - bhat);
L2norm = (Aub'*Aub)/d1/d2;
trHiAA = sum(hhat2(:).*M(:));

y = -d1*d2*sigma^2 + L2norm + 2*sigma^2*trHiAA;
