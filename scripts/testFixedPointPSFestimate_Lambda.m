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
[h,hhat] = makeGausPSF([d1,d2],omegaX,omegaY,theta);
bhat = fft2(b);

% set options
opts.sigma = sigma;
opts.V = V;
opts.tol = 1e-4;
opts.iter = 200;
[U,out] = SURE_FP_Lambda(bhat,hhat,opts);


figure(781);tiledlayout(2,2);colormap(gray);
nexttile;
imagesc(U,[0 1]);
nexttile;
imagesc(b,[0 1]);
nexttile;semilogy(out.lambdas);
nexttile;
semilogy(out.UPREs);