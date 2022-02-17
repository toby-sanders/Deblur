clear;
d1 = 128;
d2 = 128;
theta = 25;
sigmaX = 1;
sigmaY = 3;
del = 1e-7;

[h,DhX,DhY,DhT] = makeGausPSF_AndDer2DAngled([d1,d2],sigmaX,sigmaY,theta);
% [h2,DhX2,DhY2] = makeGausPSF_AndDer2D([d1,d2],sigmaX,sigmaY,1);
h2 = makeGausPSF_AndDer2DAngled([d1,d2],sigmaX,sigmaY,theta+del*180/pi);
h3 = makeGausPSF_AndDer2DAngled([d1,d2],sigmaX+del,sigmaY,theta);
h4 = makeGausPSF_AndDer2DAngled([d1,d2],sigmaX,sigmaY+del,theta);

DhT2 = (h2-h)/(del);
DhX2 = (h3-h)/del;
DhY2 = (h4-h)/del;


figure(199);
subplot(2,2,1);imagesc(fftshift(DhY));colorbar;
subplot(2,2,2);imagesc(fftshift(DhT));colorbar;
subplot(2,2,4);imagesc(fftshift(DhT2));colorbar;
subplot(2,2,3);imagesc(fftshift(DhY2));colorbar;


myrel(DhT,DhT2)
myrel(DhX,DhX2)
myrel(DhY,DhY2)