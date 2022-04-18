d1 = 129;
d2 = 129;
omegaX = .5;
omegaY = 1;
theta = 20;
pixelSize = 0.1;


[h,hhat] = makeGausPSF([d1,d2],omegaX,omegaY,theta,pixelSize);
[g,ghat] = makeGausPSFnew([d1,d2],omegaX,omegaY,theta,pixelSize);


myrel(fft2(h),hhat)
myrel(fft2(g),ghat)
myrel(g,h)