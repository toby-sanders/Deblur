clear;
K = 2;
omega = 1;

path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\';
% path = '/home/tobysanders/Dropbox/archives/data/testImages/';
imName = 'lena.png';
I = im2double(rgb2gray(imread([path,imName])));

[d1,d2] = size(I);
[h,hhat] = makeGausPSF([d1,d2],omega,omega,0,1);



% blur kernel for downsampling
g = zeros(d1,d2);
g([1:K],[1:K]) = 1/K^2;
g = fraccircshift(g,[-K/2 + 1/2, -K/2 + 1/2]);
ghat = fft2(g);


A = getSuperResDeblurOpers(d1,d2,K,hhat,ghat);


f1 = ifft2(fft2(I).*ghat.*hhat);
f1 = f1(1:K:end,1:K:end);
f2 = reshape(A(I,1),d1/K,d2/K);


myrel(f1,f2)