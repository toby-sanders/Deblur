% testing fixed point methods for optimizing a PSF and regularization
% parameters using SURE
clear;
SNR = 20;
omegaX = 1; % standard deviation of initial PSF
omegaY = 2;
theta = 20;
order = 2;
levels = 1;
rng(2021);
Ttheta = linspace(-85,85,40);
clip = 20;
pad = 20;



% get image and add noise
path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\';
% path = '/home/tobysanders/Dropbox/archives/data/testImages/';
I = im2double(rgb2gray(imread([path,'lena.png'])));
% I = im2double((imread([path,'house.tif'])));I = I(:,:,1);
% I = im2double(rgb2gray(imread([path,'peppers2.png'])));
% I = im2double((imread([path,'confFlag.tif'])));
% I = im2double(rgb2gray(imread([path,'PineyMO_Oimage.tif'])));
% I = im2double(rgb2gray(imread([path,'monarch.png'])));
% I = im2double((imread('cameraman.tif')));
% I = phantom(512);
% I = im2double(rgb2gray(imread([path,'IMG_1502.jpeg'])));

% make PSF and blurry/noisy image data
[d1,d2] = size(I);
[g,ghat] = makeGausPSF([d1,d2],omegaX,omegaY,theta);
b = real(ifft2(fft2(I).*ghat));
[b,sigma] = add_Wnoise(b,SNR);


b = b(clip+1:end-clip,clip+1:end-clip);
I = I(clip+1:end-clip,clip+1:end-clip);
b = padEdgesColorIm(b,pad);
I = padEdgesColorIm(I,pad);

[d1,d2] = size(I);

V = my_Fourier_filters(order,levels,d1,d2,1);
bhat = fft2(b);

% set options
opts.sigma = sigma;
opts.V = V;
opts.tol = 1e-4;
opts.iter = 200;

% run the fixed point iteration over all parameters
outT = SURE_FP_PSF3D(bhat,(omegaX+omegaY)/2,opts);
Niter = numel(outT.omegasX);
thetaR = outT.thetas(end);
omegaXR = outT.omegasX(end);
omegaYR = outT.omegasY(end);
gopts.theta = thetaR;
[h,hhat] = makeGausPSF([d1,d2],omegaXR,omegaYR,thetaR);

% Wiener solution based on true PSF
[g,ghat] = makeGausPSF([d1,d2],omegaX,omegaY,theta);
Ubest = real(ifft2(bhat.*conj(ghat)./(abs(ghat).^2 + outT.lambdas(end)*V)));
BMopts.profile = 'default';
UBM = GBM3D_deconv(b,outT.hhat,sigma,BMopts);
UBM2 = GBM3D_deconv(b,ghat,sigma,BMopts);


% removing the extra padding from all images
I = RMpadEdgesColorIm(I,pad);
b = RMpadEdgesColorIm(b,pad);
outT.U = RMpadEdgesColorIm(outT.U,pad);
Ubest = RMpadEdgesColorIm(Ubest,pad);
UBM = RMpadEdgesColorIm(UBM,pad);
UBM2 = RMpadEdgesColorIm(UBM2,pad);

%% display
gammaP = .5;
wP = 10;
figure(791);tiledlayout(2,3);colormap(jet);
t1 = nexttile;hold off;
plot(outT.omegasX);hold on;
plot(outT.omegasY);
plot(1:Niter,omegaX*ones(1,Niter),'k--');
plot(1:Niter,omegaY*ones(1,Niter),'k--');
title('omega convergence (with theta)');
legend('omegaX','omegaY');
hold off;
t1 = nexttile;
plot((outT.thetas));title('theta convergence');
t1 = nexttile;
semilogy(outT.lambdas);title('lambda convergence');
t2 = nexttile;
imagesc(fftshift(g));title('true PSF');colorbar;
t3 = nexttile;
imagesc(fftshift(h));title('recovered PSF');colorbar;
linkaxes([t3 t2]);
axis([d2/2-wP+1, d2/2 + wP, d1/2-wP+1, d1/2 + wP]);

figure(792);tiledlayout(2,3);colormap(gray);
t4 = nexttile;imagesc(b,[0 1]);title(sprintf('blurry/noisy image, PSNR: %g',myPSNR(I,b,1)));
t5 = nexttile;imagesc(outT.U,[0 1]);title(sprintf('recovered image, PSNR: %g',myPSNR(I,outT.U,1)));
t6 = nexttile;imagesc(Ubest,[0 1]);title(sprintf('ideal recovered, PSNR: %g',myPSNR(I,Ubest,1)));
t7 = nexttile;imagesc(UBM,[0 1]);title(sprintf('BM3D deblur, PSNR: %g',myPSNR(I,UBM,1)));
t8 = nexttile;imagesc(UBM2,[0 1]);title(sprintf('ideal BM3D deblur, PSNR: %g',myPSNR(I,UBM2,1)));
linkaxes([t4 t5 t6 t7 t8]);

% figure(793);hold off;
% semilogy(abs(outT.tmp1(3:end)));hold on;
% semilogy(real(abs(outT.tmp2(3:end))));hold off;
% title('theta gradient terms');
% xlabel('iteration');