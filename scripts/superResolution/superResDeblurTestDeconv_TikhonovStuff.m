clear;
d =  256; % dimension of downsampled vector
K = 2;
omega = 2; % blurring width
SNR = 100; % SNR in data


% simulate data
% x = phantom(d0);
path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\';
% path = '/home/tobysanders/Dropbox/archives/data/testImages/';
imName = 'lena.png';
x = im2double(rgb2gray(imread([path,imName])));

[d1,d2] = size(x);



% original blur
[h,hhat] = makeGausPSF([d1,d2],omega);
% blur kernel for downsampling
g = zeros(d1,d2);
g([1:K],[1:K]) = 1/K^2;
g = fraccircshift(g,[-K/2 + 1/2, -K/2 + 1/2]);
ghat = fft2(g);



V = my_Fourier_filters(1,1,d1,d2,1);


b = ifft2(ghat.*hhat.*fft2(x));
b = b(1:K:end,1:K:end);
[b,sigma] = add_Wnoise(b,SNR);
bstr = zeros(d1,d2);
bstr(1:K:end,1:K:end) = b;



A = getSuperResDeblurOpers(d1,d2,K,hhat,ghat);
[~,~,scl] = ScaleA(d1*d2,A,b(:));
tik.order = 1;
tik.mu = SNR*1e1;
tik.scale_A = false;
tik.iter = 200;
[rec1,out1] = Tikhonov(A,b(:),[d1,d2,1],tik);
[rec2,out] = Tikhonov_SUPER(b,K,hhat,tik);
rec2 = real(rec2);

%%
out1.total_time
out.total_time

figure(712);tiledlayout(2,3);colormap(gray);
t1 = nexttile;imagesc(rec1,[0 1]);
t2 = nexttile;imagesc(rec2,[0 1]);
t3 = nexttile;imagesc(x,[0 1]);
t4 = nexttile;semilogy(real(out.rel_chg));
linkaxes([t1 t2 t3]);