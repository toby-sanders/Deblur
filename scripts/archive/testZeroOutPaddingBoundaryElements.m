% testing fixed point methods for optimizing a PSF and regularization
% parameters using SURE
clear;
SNR = 200;
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
path = 'C:\Users\Toby Sanders\Dropbox\archives\data\testImages\';
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
[d1,d2] = size(I);



I2 = zeros(d1+pad,d2+pad);
b2 = zeros(d1+pad,d2+pad);
b2(1:d1,1:d2) = b;
I2(1:d1,1:d2) = I;
mask = b2;
mask(1:d1,1:d2) = 1;

[d1,d2] = size(mask);
[h,hhat] = makeGausPSF([d1,d2],omegaX,omegaY,theta);

V = my_Fourier_filters(order,levels,d1,d2,1);
bhat = fft2(b);

A = @(x,mode)deconvOperZeroPad(x,mode,hhat,mask);


tik.mu = 1e3;
tik.order = 2;
tik.iter = 250;
[rec1,out1] = Tikhonov(A,b2(:),[d1,d2,1],tik);


mask2 = mask;
mask2(:) = 1;
A2 = @(x,mode)deconvOperZeroPad(x,mode,hhat,mask2);
[rec2,out2] = Tikhonov(A2,b2(:),[d1,d2,1],tik);

%%
mmF = [0 1];
figure(901);tiledlayout(2,2);colormap(gray);
t1 = nexttile;
imagesc(real(rec1),mmF);colorbar;
t2 = nexttile;
imagesc(real(rec2),mmF);colorbar;
t3 = nexttile;
imagesc(b2,mmF);colorbar;
linkaxes([t2 t1 t3])






