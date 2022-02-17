clear;
K = 2;
omega = 1; % blurring width
SNR = 50; % SNR in data
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



nIm = randn(size(b))*sigma;
nImStr = zeros(d1,d2);
nImStr(1:K:end,1:K:end) = nIm;

% u3 is done by approximating matrix inverse with diagonal
lambda = 10/SNR^2; % reg. parameter
filt = conj(hhat).*conj(ghat)./(hhat.*conj(hhat).*conj(ghat).*ghat/K^2 + lambda*V);
u3 = real(ifft2(fft2(bstr).*filt));
sigmaPSD = sigma^2*abs(filt).^2;
BMopts.profile = 'default';

fudgeFactor = 3e-1;
u4 = GBM3D(u3,sigmaPSD*fudgeFactor,BMopts);


% "noise" reconstructions
alphaRI = 250*ones(d1,d2)*sigma^2;
regMatrix = 1e-1*V.*alphaRI;
[nRec,outTik] = Tikhonov_SUPER(nIm,K,hhat,regMatrix);
nRec2 = real(ifft2(fft2(nImStr).*filt));





[u5,out] = GBM3D_SUPER(b,K,hhat,sigma,BMopts);
u6 = Tikhonov_SUPER(b,K,hhat,regMatrix);


FN = fftshift(abs(fft2(nRec)).^2);
FN = imfilter(FN,ones(4)/16);
FN2 = fftshift(abs(fft2(nRec2)).^2);
FN2 = imfilter(FN2,ones(4)/16);


%%
dataStr = imresize(b,K,'nearest');
figure(217);colormap(gray);tiledlayout(2,3);
t1 = nexttile;
imagesc(imresize(b,K,'nearest'),[0 1]);
title(sprintf('blurry im: PSNR = %g',myPSNR(x,dataStr,1)));
t2 = nexttile;
imagesc(x);title('original');
t3 = nexttile;
imagesc(u4,[0 1]);
title(sprintf('Wiener + BM3D: PSNR = %g',myPSNR(x,u4,1)));
t5 = nexttile;
imagesc(u5,[0 1]);
title(sprintf('BM3D super res func. = %g',myPSNR(x,u5,1)));
t6 = nexttile;
imagesc(u3,[0 1]);title('wiener solution passed to BM3D')
% imagesc(u6,[0 1]);title('Wiener solution');
t7 = nexttile;
imagesc(out.recWie,[0 1]);title('Wiener solution from BM3D function');
% t4 = nexttile;
% imagesc(p3rec,[0 1]);
% title(sprintf('P3 rec: PSNR = %g',myPSNR(x,p3rec,1)));
linkaxes([t1 t2 t3 t5 t6 t7]);





figure(912);tiledlayout(3,3);
v1 = nexttile;imagesc(FN);colorbar;title('noise PSD from iterative Tik. reg. sol.')
v2 = nexttile;imagesc(fftshift(sigmaPSD));colorbar;title('PSD from constructed deconv. filter')
v3 = nexttile;imagesc(FN2);colorbar;title('PSD from filtered noise');
v4 = nexttile;imagesc(nRec);colorbar;title('noise after iterated Tik. sol.')
v5 = nexttile;imagesc(nRec2);colorbar;title('noise after filtering');
linkaxes([v1 v2 v3]);
linkaxes([v4 v5]);