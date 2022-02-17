clear;
d = 512;
omegaX = 6;
omegaY = 1;
theta = 20;
SNR = 20;
lambdas = linspace(4,-2,30); % test values for lambda (for comparison only)
lambdas = 10.^(-lambdas);

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


[d1,d2] = size(I);

% generate PSF, shift, and then rotate
[h,hhat] = makeGausPSF2D([d1,d2],omegaX,omegaY);
h = fftshift(h);
hhat = fftshift(hhat);
h = imrotate(h,theta,'bilinear','crop');
hhat = imrotate(hhat,theta,'bilinear','crop');

% fft and ifft of rotated PSF and FPSF
h2 = real(fftshift(ifft2(ifftshift(hhat))));
hhat2 = real(fftshift(fft2(ifftshift(h))));


ghat = fft2(ifftshift(h));
ghat2 = ifftshift(hhat2);
[~,ghat3] = makeGausPSF2D([d1,d2],omegaX,omegaY);
b = ifft2(fft2(I).*ghat);
[b,sigma] = add_Wnoise(b,SNR);
V = my_Fourier_filters(2,1,d1,d2,1);
bhat = fft2(b);

opts.sigma = sigma;
opts.V = V;
opts.lambdas = lambdas;

out1 = SURE_deblur(bhat,ghat,opts);
out2 = SURE_deblur(bhat,ghat2,opts);
out3 = SURE_deblur(bhat,ghat3,opts);

%%
figure(987);tiledlayout(2,3);colormap(gray);
s1 = nexttile;hold off;
plot(out1.SURE);hold on;
plot(out2.SURE);
plot(out3.SURE);
title('SURE comparisons');
legend('correct PSF','real/real PSF','non-angled PSF');
hold off;
s2 = nexttile;imagesc(real(out1.rBest),[0 1]);title('recovered, exact PSF');
s3 = nexttile;imagesc(real(out2.rBest),[0 1]);title('recovered, slightly wrong PSF, with all real parts')
s4 = nexttile;imagesc(imag(out1.rBest));colorbar;title('imaginary part of rec');
s5 = nexttile;imagesc(imag(out2.rBest));colorbar;title('imaginary part of rec');
s6 = nexttile;imagesc(b);title('blurry');


myrel(out1.rBest,I)
myrel(out2.rBest,I)






