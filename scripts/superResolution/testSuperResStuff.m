% basic small example testing super resolution Wiener filter
% in summary, if the blur is very small (sigma<<1) and/or lambda is very
% small (lambda<<1), or some combination of those two things, then the
% diagonal Wiener filter approximation is poor. In all other instances is
% works suitably well.


clear;
d = 32;
K = 2;
N = K*d;
sigma = 1;
lambda = 1e-1;

% construct row sampling matrix and unitary Fourier operator
P = zeros(N,N);
for i = 1:d
    P((i-1)*K+1,(i-1)*K+1) = 1;
end
v = 0:N-1;
F = exp(1i*2*pi*v'*v/N)/sqrt(N);



% construct all of the convolutional operators and reg. matrices
A = F*P'*P*F';
[h,hhat] = makeGausPSF1D(N,sigma);
g = zeros(N,1);
g(1:K) = 1/K;
g = fraccircshift(g,-round(K/2)+1/2);
ghat = fft2(g);
T = zeros(N);
T(1:N+1:end) = -1;
T(N+1:N+1:end) = 1;
T(N,1) = 1;
That = F'*T'*T*F;
A2 = eye(N)/K;
Hhat = zeros(N);
Hhat(1:N+1:end) = hhat;
Ghat = Hhat;
Ghat(1:N+1:end) = ghat;


% this is the matrix inverse needed in the Wiener filter
% "M" is the exact one, and M2 is the pragmatic alternative
M = conj(Hhat*Ghat)*A*Hhat*Ghat + lambda*That;
M2 = conj(Hhat*Ghat)*A2*Hhat*Ghat + lambda*That;

fprintf('diff. between approx. matrix and exact matrix: %g\n',myrel(M,M2));


figure(77);subplot(2,2,1);
imagesc(real(F'*F));colorbar;
title('sanity check for Four. trans.')
subplot(2,2,2);
imagesc(real(M));colorbar;
title('exact matrix needed to invert')
subplot(2,2,3);
imagesc(real(A));colorbar;
title('matrix with subsampling stuff')
subplot(2,2,4);
imagesc(real(M2));colorbar;
title('approx of matrix needed to invert')