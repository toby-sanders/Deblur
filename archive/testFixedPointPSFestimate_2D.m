% testing fixed point methods for optimizing a PSF and regularization
% parameters using SURE
clear;
SNR = 20;
omegaX = 3; % standard deviation of initial PSF
omegaY = .1;
order = 2;
levels = 1;
rng(2021);


% get image and add noise
path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\';
% path = '/home/tobysanders/Dropbox/archives/data/testImages/';
I = im2double(rgb2gray(imread([path,'lena.png'])));
% I = im2double((imread([path,'house.tif'])));I = I(:,:,1);
% I = im2double(rgb2gray(imread([path,'peppers2.png'])));
% I = im2double((imread([path,'confFlag.tif'])));
% I = im2double((imread([path,'SeasatDeblurred.png'])));
% I = im2double(rgb2gray(imread([path,'PineyMO_Oimage.tif'])));
% I = im2double(rgb2gray(imread([path,'monarch.png'])));
% I = im2double((imread('cameraman.tif')));
% I = phantom(512);

% make PSF and blurry/noisy image data
[d1,d2] = size(I);
[g,ghat] = makeGausPSF2D([d1,d2],omegaX,omegaY);
b = real(ifft2(fft2(I).*ghat));
[b,sigma] = add_Wnoise(b,SNR);
V = my_Fourier_filters(order,levels,d1,d2,1);
bhat = fft2(b);

% set options
opts.sigma = sigma;
opts.V = V;
opts.tol = 1e-7;
opts.iter = 250;

% initialize some random PSF variance
omega0 = (omegaX + omegaY)/2;

% reset lambda and run FP over both PSF and lambda
opts.lambda = 8.4e2*sigma^2;
out = SURE_FP(bhat,omega0,opts);
out2 = SURE_FP2(bhat,omega0,opts);

%%
Niter = numel(out2.omegasY);
figure(901);
subplot(2,2,1);hold off;
plot(out.omegas);hold on;
plot(1:Niter,omegaY*ones(1,Niter),'--k');
plot(1:Niter,omegaX*ones(1,Niter),'--k');
legend('recovered omega','True omegas');
title('single omega search');hold off;

subplot(2,2,2);hold off;
plot(out2.omegasX);hold on;
plot(out2.omegasY);
plot(1:Niter,omegaY*ones(1,Niter),'--k');
plot(1:Niter,omegaX*ones(1,Niter),'--k');
legend('X omegas','Y omegas','True omegas');
hold off;

subplot(2,2,3);hold off;
plot(real(out.lambdas));hold on;
plot(real(out2.lambdas));
title('lambda convergence');
legend('single omega','double omega');

figure(902);tiledlayout(2,2);colormap(gray);
t1 = nexttile;imagesc(out.U,[0 1]);
title(sprintf('single omega, PSNR = %g',myPSNR(I,out.U,1)));
t2 = nexttile;imagesc(out2.U,[0 1]);
title(sprintf('double omega, PSNR = %g',myPSNR(I,out2.U,1)));
t3 = nexttile;imagesc(b,[0 1]);
title(sprintf('blurry/noisy, PSNR = %g',myPSNR(I,b,1)));
linkaxes([t1 t2 t3]);