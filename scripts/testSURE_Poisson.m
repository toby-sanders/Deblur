clear;
% mu0 = 100;
SNR = 20;
omegaX = 2; % standard deviation of initial PSF
omegaY = 1;
theta = 20;
order = 2;
levels = 1;
rng(2021);

mu0 = SNR^2;
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


I = I*mu0;

% make PSF and blurry/noisy image data
[d1,d2] = size(I);
[g,ghat] = makeGausPSF([d1,d2],omegaX,omegaY,theta);
b = real(ifft2(fft2(I).*ghat));
sigma = sqrt(mu0)*mean(b(:));
b = imnoise(b*1e-12,'poisson')*1e12;
sigma = mean(sqrt(b(:)));
% b = b + randn(size(b))*sigma;

estimateScalePois(b*750);


V = my_Fourier_filters(order,levels,d1,d2,1);
bhat = fft2(b);

% set options
opts.sigma = sigma;
opts.V = V;
opts.tol = 1e-4;
opts.iter = 200;

% run the fixed point iteration over all parameters
out0 = SURE_FP_PSF3D(bhat,(omegaX+omegaY)/2,opts);
outT = SURE_Pois_FP_PSF3D(bhat,(omegaX+omegaY)/2,opts);
Niter = numel(outT.omegasX);
thetaR = outT.thetas(end);
omegaXR = outT.omegasX(end);
omegaYR = outT.omegasY(end);
gopts.theta = thetaR;
[h,hhat] = makeGausPSF([d1,d2],omegaXR,omegaYR,thetaR);

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
nexttile;
hold off;
plot(outT.UPREs);
title('SURE values');

figure(792);tiledlayout(2,3);colormap(gray);
t4 = nexttile;imagesc(b,[0 max(I(:))]);title(sprintf('blurry/noisy image, PSNR: %g',myPSNR(I,b)));
t5 = nexttile;imagesc(outT.U,[0 max(I(:))]);title(sprintf('recovered image, Pois PSNR: %g',myPSNR(I,outT.U)));
t6 = nexttile;imagesc(out0.U,[0 max(I(:))]);title(sprintf('recovered image, Gaus PSNR: %g',myPSNR(I,out0.U)));
% t6 = nexttile;imagesc(Ubest,[0 1]);title(sprintf('ideal recovered, PSNR: %g',myPSNR(I,Ubest,1)));
% t7 = nexttile;imagesc(UBM,[0 1]);title(sprintf('BM3D deblur, PSNR: %g',myPSNR(I,UBM,1)));
% t8 = nexttile;imagesc(UBM2,[0 1]);title(sprintf('ideal BM3D deblur, PSNR: %g',myPSNR(I,UBM2,1)));
linkaxes([t4 t5 t6]);