% testing fixed point methods for optimizing a PSF and regularization
% parameters using SURE
clear;
SNR = 60;
omegaX = 1; % standard deviation of initial PSF
omegaY = 3;
theta = 25;
order = 2;
levels = 1;
pad = 100;
rng(2021);



% get image and add noise
path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\';
I = im2double(rgb2gray(imread([path,'lena.png'])));


[d1,d2] = size(I);
[g,ghat] = makeGausPSF([d1,d2],omegaX,omegaY,theta);
b = real(ifft2(fft2(I).*ghat));
[b,sigma] = add_Wnoise(b,SNR);


V = my_Fourier_filters(order,levels,d1,d2,1);
bhat = fft2(b);

% set options
opts.sigma = sigma;
opts.V = V;
opts.tol = 1e-4;
opts.iter = 200;

% run the fixed point iteration over all parameters
tic;
out = SURE_FP_PSF3D(bhat,(omegaX+omegaY)/2,opts);
toc;
Niter = numel(out.omegasX);
thetaR = out.thetas(end);
omegaXR = out.omegasX(end);
omegaYR = out.omegasY(end);




fun = @(x)SUREobj(x,d1,d2,bhat,V,sigma);
x0 = [1 1 0 8.4e2*sigma^2];
tic;
[xR,fVal,exitFlag,output] = fminsearch(fun,x0);
toc;

%%
figure(428);
subplot(2,2,1);hold off;
plot(out.omegasX);hold on;
plot(xR(1)*ones(Niter,1),'--');
plot(ones(Niter,1)*omegaX,':k');hold off;
subplot(2,2,2);hold off;
plot(out.omegasY);hold on;
plot(xR(2)*ones(Niter,1),'--');
plot(ones(Niter,1)*omegaY,':k');hold off;
subplot(2,2,3);hold off;
loglog(out.UPREs);hold on;
loglog(fVal*ones(Niter,1));hold off;
subplot(2,2,4);hold off;
plot(out.thetas);hold on;
plot(ones(Niter,1)*xR(3)*180/pi);
plot(ones(Niter,1)*theta,':k');hold off;










