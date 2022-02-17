% testing fixed point methods for optimizing a PSF and regularization
% parameters using SURE
clear;
SNR = 50;
omega = 2;
zeta = 3;
alpha = .8;
beta = 1-alpha;
order = 2;
levels = 1;
rng(2021);


% get image and add noise
path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\';
% path = '/home/tobysanders/Dropbox/archives/data/testImages/';
% I = im2double(rgb2gray(imread([path,'lena.png'])));
% I = im2double((imread([path,'house.tif'])));I = I(:,:,1);
% I = im2double(rgb2gray(imread([path,'peppers2.png'])));
% I = im2double((imread([path,'confFlag.tif'])));
% I = im2double((imread([path,'SeasatDeblurred.png'])));
% I = im2double(rgb2gray(imread([path,'PineyMO_Oimage.tif'])));
I = im2double(rgb2gray(imread([path,'monarch.png'])));
% I = im2double((imread('cameraman.tif')));
% I = phantom(512);

% make PSF and blurry/noisy image data
[d1,d2] = size(I);
[g,ghat] = makeGausPSF([d1,d2],omega,omega,0,1);
[f,fhat] = makeMotionPSF2D([d1,d2],zeta,zeta,0,'tophat');
h = alpha*g + beta*f;
fhat = real(fhat);
hhat = alpha*ghat + beta*fhat;
b = real(ifft2(fft2(I).*hhat));
[b,sigma] = add_Wnoise(b,SNR);
V = my_Fourier_filters(order,levels,d1,d2,1);
bhat = fft2(b);

% set options
opts.sigma = sigma;
opts.V = V;
opts.tol = 1e-7;
opts.iter = 250;


% reset lambda and run FP over both PSF and lambda
opts.lambda = 8.4e2*sigma^2;
[U,out] = SURE_FP_Lambda(bhat,hhat,opts);
% out2 = SURE_FP_PSF2D_twoPSFs(bhat,ghat,fhat,opts);
out2 = SURE_FP_PSF_multiPSF(bhat,cat(3,ghat,fhat),opts);
hR = ifft2(out2.hhat);
%%
Niter = numel(out2.alphas);
figure(901);
subplot(3,3,1);hold off;
plot(out2.alphas(1,:));hold on;
plot(out2.alphas(2,:));
plot(1:Niter,alpha*ones(1,Niter),':b');
plot(1:Niter,beta*ones(1,Niter),':r');
legend('alphas','betas','True alpha','true beta');
axis([0 Niter,-.1,1.1])
hold off;

subplot(3,3,2);hold off;
plot(real(out.lambdas));hold on;
plot(real(out2.lambdas));
title('lambda convergence');
legend('lambda only','alpha/beta');

subplot(3,3,3);imagesc(fftshift(h));colorbar;
axis([d2/2-15,d2/2+15,d1/2-15,d1/2+15]);
title('true Psf');
subplot(3,3,4);imagesc(fftshift(hR));colorbar;
axis([d2/2-15,d2/2+15,d1/2-15,d1/2+15]);
title('recovered psf');
subplot(3,3,5);hold off;
semilogy(real(out.UPREs));hold on;
semilogy(out2.UPREs);hold off;
title('UPRE values');xlabel('iteration');
legend('lambda only','alpha/beta');
subplot(3,3,6);plot(out2.normer);
title('normalization needed');
xlabel('iteration');

figure(902);tiledlayout(2,2);colormap(gray);
t1 = nexttile;imagesc(U,[0 1]);
title(sprintf('lambda only FP, PSNR = %g',myPSNR(I,U,1)));
t2 = nexttile;imagesc(out2.U,[0 1]);
title(sprintf('alpha/beta FP, PSNR = %g',myPSNR(I,out2.U,1)));
t3 = nexttile;imagesc(b,[0 1]);
title(sprintf('blurry/noisy, PSNR = %g',myPSNR(I,b,1)));
linkaxes([t1 t2 t3]);