clear;
d = 128;
opts.theta = 75.2698;
opts.pixelSize = 1;
sigmaX = 3;
sigmaY = 6;


% [h,hhat] = makeGausPSF([d,d],sigmaX,sigmaY,opts);
% [g,ghat] = makeGausPSF2DAngled([d,d],sigmaX,sigmaY,opts.theta);
[h,hhat,DhX,DhY,DhT] = makeGausPSFAndDer([d,d],sigmaX,sigmaY,opts.theta,opts.pixelSize);
[hhat2,DhX2,DhY2,DhT2] = makeGausPSF_AndDer2DAngled([d,d],sigmaX,sigmaY,opts.theta);

figure(777);subplot(3,2,1);imagesc(fftshift(DhX));
subplot(3,2,2);imagesc(fftshift(DhX2));
subplot(3,2,3);imagesc(fftshift(DhY))
subplot(3,2,4);imagesc(fftshift(DhY2));
subplot(3,2,5);imagesc(fftshift(DhT));
subplot(3,2,6);imagesc(fftshift(DhT2));


myrel(DhX,DhX2)
myrel(DhY,DhY2)
myrel(DhT,DhT2)