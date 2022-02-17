% basic small example testing super resolution Wiener filter
% this particular example considers the blur operator, h, to take place
% after the downsampling/binning instead of before. It appears that the
% diagonal approximation to the Wiener filter is roughly the same either
% way
% this is a semi-important bit, since for super-resolution where the blur
% is estimated on the original low resolution image, then the blur is
% essentially considered to take place after the binning/downsampleing..


clear;
d = 32;
K = 2;
N = K*d;
sigma = 1;
lambda = 1e-1;

% construct row sampling matrix and unitary Fourier operator
P = zeros(d,N);
for i = 1:d
    P(i,2*i-1) = 1;
end
v = 0:N-1;
F = exp(1i*2*pi*v'*v/N)/sqrt(N);
v2 = 0:d-1;
F2 = exp(1i*2*pi*v2'*v2/d)/sqrt(d);


% construct all of the convolutional operators and reg. matrices
A = F2*P*F';
A2 = F*P'*F2';

[h,hhat] = makeGausPSF1D(d,sigma);
g = zeros(N,1);
g(1:K) = 1/K;
g = fraccircshift(g,-round(K/2)+1/2);
ghat = fft2(g);
T = zeros(N);
T(1:N+1:end) = -1;
T(N+1:N+1:end) = 1;
T(N,1) = 1;
That = F'*T'*T*F;
% A2 = eye(N)/K;
Hhat = zeros(d);
Hhat(1:d+1:end) = hhat;
Ghat = zeros(N);
Ghat(1:N+1:end) = ghat;

Hhat2 = zeros(N,d);
Hhat2(1:d,1:d) = Hhat;
Hhat2(d+1:end,1:d)  = Hhat;
Hhat3 = zeros(N);
Hhat3(1:d,1:d) = Hhat*Hhat;
Hhat3(d+1:end,d+1:end) = Hhat*Hhat;


% this is the matrix inverse needed in the Wiener filter
% "M" is the exact one, and M2 is another close alternate
% M3 is the pragmatic diagonal alternative
M = conj(Ghat)*A2*conj(Hhat)*Hhat*A*Ghat + lambda*That;
M2 = conj(Ghat)*conj(Hhat2)*Hhat2'*Ghat/K + lambda*That;
M3 = conj(Ghat)*Hhat3*Ghat/K + lambda*That;
myrel(M,M3)

figure(901);
subplot(2,2,1);imagesc(real(A));colorbar;title('row sample after fourier transforms')
subplot(2,2,2);imagesc(real(M3));colorbar;title('diagonal estimate');
subplot(2,2,3);imagesc(real(M));colorbar;title('true matrix that needs inverting')
subplot(2,2,4);imagesc(real(M2));colorbar;title('another estimate')