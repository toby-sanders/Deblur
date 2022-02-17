clear;
K = 2;
omega = 0.7; % blurring width
SNR = 25; % SNR in data
rng(2022);

% simulate data
% x = phantom(d0);
path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\';
% path = '/home/tobysanders/Dropbox/archives/data/testImages/';
imName = 'lena.png';
x = im2double(rgb2gray(imread([path,imName])));
% x = im2double(imread('cameraman256.png'));
% x = x(1:1024,1:1024);
[d1,d2] = size(x);

% trim image if dimensions are not even
if mod(d1,2)~=0
    x = x(1:end-1,:);
end
if mod(d2,2)~=0
    x = x(:,1:end-1);
end
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






% u3 is done by approximating matrix inverse with diagonal
lambda = 20/SNR^2; % reg. parameter
filt = conj(hhat).*conj(ghat)./(hhat.*conj(hhat).*conj(ghat).*ghat/K^2 + lambda*V);
u3 = real(ifft2(fft2(bstr).*filt));
sigmaPSD = sigma^2*abs(filt).^2;
BMopts.profile = 'default';

fudgeFactor = 3e-1;
u4 = GBM3D(u3,sigmaPSD*fudgeFactor,BMopts);

% A = getSuperResDeblurOpers(d1,d2,K,hhat,ghat);
% Reg = getGBM3D;
% popts.sigma0 = sigma*0.50;% 1/255;
% popts.iter = 25;
% popts.tol = 1e-5;
% popts.init = u3;
% [p3rec,p3out] = PnP3_prox(A,Reg,b(:),[d1,d2,1],popts);
[u5,out] = GBM3D_SUPER(b,K,hhat,sigma,BMopts);



%%
dataStr = imresize(b,K,'nearest');
figure(217);colormap(gray);tiledlayout(2,2);
t1 = nexttile;
imagesc(imresize(b,K,'nearest'),[0 1]);
title(sprintf('blurry im: PSNR = %g',myPSNR(x,dataStr,1)));
t2 = nexttile;
imagesc(x);
t3 = nexttile;
imagesc(u4,[0 1]);
title(sprintf('Wiener + BM3D: PSNR = %g',myPSNR(x,u4,1)));
t5 = nexttile;
imagesc(u5,[0 1]);
title(sprintf('BM3D super res func. = %g',myPSNR(x,u5,1)));
% t4 = nexttile;
% imagesc(p3rec,[0 1]);
% title(sprintf('P3 rec: PSNR = %g',myPSNR(x,p3rec,1)));
linkaxes([t1 t2 t3 t5]);