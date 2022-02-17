d = 129;
d2 = 350;
alphaX = 30;
alphaY = 3;
theta = 20;
delta = 1e-7;

[~,hhat,Dx,Dy,DT] = makeLaplacePSFAndDer([d,d2],alphaX,alphaY,theta);
[~,hhatX] = makeLaplacePSFAndDer([d,d2],alphaX+delta,alphaY,theta);
[~,hhatY] = makeLaplacePSFAndDer([d,d2],alphaX,alphaY+delta,theta);
[~,hhatT] = makeLaplacePSFAndDer([d,d2],alphaX,alphaY,theta+delta);

Dx2 = (hhatX-hhat)/delta;
Dy2 = (hhatY-hhat)/delta;
DT2 = (hhatT-hhat)/(delta*pi/180);

myrel(Dx,Dx2)
myrel(Dy,Dy2)
myrel(DT,DT2)

figure(777);tiledlayout(2,4);
t1 = nexttile;
imagesc(fftshift(hhat));colorbar;
t2 = nexttile;
imagesc(fftshift(Dx));colorbar;
t3 = nexttile;
imagesc(fftshift(Dy));colorbar;
t4 = nexttile;
imagesc(fftshift(DT));colorbar;
t5 = nexttile;imagesc(fftshift(real(ifft2(hhat))));colorbar;
t6 = nexttile;imagesc(fftshift(Dx2));colorbar;
t7 = nexttile;imagesc(fftshift(Dy2));colorbar;
t8 = nexttile;imagesc(fftshift(DT2));colorbar;

