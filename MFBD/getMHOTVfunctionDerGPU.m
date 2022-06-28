function D = getMHOTVfunctionDerGPU(k,levels,p,q)
% this code gets the functional derivative for MHOTV operators, which 
% speeds up the operation since the filters are preallocated

% k is the order of the transform
% levels is the number of scales used for the FD transforms
% recommended 3 levels
%
% for k=1 and levels=1, the method is just TV

% Written by Toby Sanders @Lickenbrock Tech.
% 09/19/2019

l = 0:levels-1; l = 2.^l;
VX = gpuArray(zeros(p,q,levels)); VY = VX; 
for ii = 1:levels % wavelet like filters
    vx = (-1)^k*[0,1/l(ii)*((exp(-1i*2*pi*(1:q-1)*l(ii)/q)-1).^(k+1))./(exp(-1i*2*pi*(1:q-1)/q)-1)];
    vy = (-1)^k*[0,1/l(ii)*((exp(-1i*2*pi*(1:p-1)*l(ii)/p)-1).^(k+1))./(exp(-1i*2*pi*(1:p-1)/p)-1)];
    [VX(:,:,ii),VY(:,:,ii)] = meshgrid(vx,vy);
end
D = @(U)preallocatedMHOTV(U,k,levels,VX,VY,p,q);

function [U,AbsGrads] = preallocatedMHOTV(U,k,levels,VX,VY,p,q)
epDenom = 1e0; % padding for division

% Transform data into frequency domain along each dimension
% and allocate FD matrices for storage
X = fft(U,q,2).*VX;
Y = fft(U,p,1).*VY;
X = 2^(1-k)/levels*ifft(X,q,2); % transform back to real space
Y = 2^(1-k)/levels*ifft(Y,p,1);

% "normalization" step
AbsGrads = sqrt(X.^2 + Y.^2) + epDenom;
X = X./AbsGrads;
Y = Y./AbsGrads;

% transpose operation
% conjugate filtering for each level and dimension
X = fft(X,q,2).*conj(VX); 
Y = fft(Y,p,1).*conj(VY);
X = ifft(X,q,2); % transform filtered data back to real space
Y = ifft(Y,p,1);

% finish transpose operation by summing and constant multiplication
U = real(sum(X+Y,3))*2^(1-k)/levels;
AbsGrads = AbsGrads - epDenom;