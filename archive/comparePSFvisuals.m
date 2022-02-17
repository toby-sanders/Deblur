clear;
d = 512;
omegaX = 4;
omegaY = .1;
theta = 45;


[h,hhat] = makeGausPSF2DAngled([d],omegaX,omegaY,theta);
[g,ghat] = makeMotionPSF2D(d,omegaX*sqrt(3),omegaY*sqrt(3),-theta,'gaus');

figure(777);
subplot(2,2,1);imagesc(fftshift(h));colorbar;
axis([d/2-15,d/2+15,d/2-15,d/2+15]);
subplot(2,2,2);imagesc(fftshift(g));colorbar;
axis([d/2-15,d/2+15,d/2-15,d/2+15]);
subplot(2,2,3);imagesc(fftshift(hhat));colorbar;
subplot(2,2,4);imagesc(fftshift(real(ghat)));colorbar;