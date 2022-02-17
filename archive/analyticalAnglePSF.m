clear;
d = 64;
sigmaX = 3;
sigmaY = 1;
theta = 140;



% [h,hhat] = makeGausPSF2D([d,d],sigmaX,sigmaY);
[g,ghat,f,fhat] = makeGausPSF2DAngled([d,d],sigmaX,sigmaY,theta);

g2 = ifft2(ghat);
f2 = ifft2(fhat);

myrel(ghat,fhat)
myrel(g,f)
myrel(fft2(f),fhat)
myrel(fft2(g),ghat)



figure(659);tiledlayout(2,3);
t1 = nexttile;
imagesc(fftshift(g));title('angle PSF, method 1')
t2 = nexttile;
imagesc(fftshift(f));title('angled PSF, method 2')
t22 = nexttile;
imagesc(real(fftshift(g2)));title('ifft of FPSF, method 1')
t3 = nexttile;
imagesc(fftshift(ghat));title('angled FPSF, method1')
t4 = nexttile;
imagesc(fftshift(fhat));title('angled FPSF, method 2')
linkaxes([t1 t2]);
linkaxes([t3 t4]);


figure(658);tiledlayout(2,2);
v1 = nexttile;
imagesc(imag(g2));title('imaginary part of ifft of FPSF, method 1');
colorbar;
v2 = nexttile;
imagesc(imag(f2));title('imaginary part of ifft of FPSF, method 2');
colorbar;