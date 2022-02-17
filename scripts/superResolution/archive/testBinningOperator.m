d = 64;
K = 2;


g = zeros(d);
g(1:K,1:K) = 1/K^2;
g = fraccircshift(g,[-K/2+1/2, -K/2+1/2]);
ghat = fft2(g);

figure(77);
subplot(2,2,1);imagesc(fftshift(g));
subplot(2,2,2);imagesc(fftshift(real(ghat)));colorbar;
subplot(2,2,3);imagesc(fftshift(imag(ghat)));colorbar;