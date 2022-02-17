% testing downsamping/upsampling fourier transforms
clear;
N = 64;



X = phantom(N);
u = X(:,N/2);
Pu = zeros(2*N,1);
Pu(1:2:end) = u;
% u = sin(1*pi*(0:N-1)'/N);
Fu = fft(u,2*N);
Fu2 = fft(Pu);
Fu3 = fft(u)

myrel(Fu3,Fu2(1:N))
[Fu2(1:N),Fu2(N+1:end)]

figure(541);
subplot(2,2,1);
plot(abs(Fu));
subplot(2,2,2);
plot(abs(Fu2));
subplot(2,2,3);
plot(abs(Fu3));