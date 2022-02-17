% testing to see how estimating Lambda at standard resolution projects on
% to the best lambda for super resolution
% most of my tests indicate the scaling lambda -> lambdaNew = lambda/K^2
% is generally close to ideal....



clear;
d =  256; % dimension of downsampled vector
K = 2;
d0 = d*K;
omega = 2.5; % blurring width
SNR = 50; % SNR in data


% simulate data
% x = phantom(d0);
path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\';
% path = '/home/tobysanders/Dropbox/archives/data/testImages/';
imName = 'peppers2.png';% 'lena.png';
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
bstr = zeros(d0,d0);
bstr(1:K:end,1:K:end) = b;

Sopts.V = my_Fourier_filters(2,1,d1/K,d2/K,1);
Sopts.iter = 150;
Sopts.sigma = sigma;
[~,Fpsf] = makeGausPSF([d1/K,d2/K],omega,omega,0,K);
Ihat = fft2(b);
[U,out] = SURE_FP_Lambda(Ihat,Fpsf,Sopts);
out2 = SURE_FP_PSF3D(Ihat,1,Sopts);


% u3 is done by approximating matrix inverse with diagonal
lambda = 70/SNR^2; % reg. parameter
filt = conj(hhat).*conj(ghat)./(hhat.*conj(hhat).*conj(ghat).*ghat/K^2 + lambda*V);
u3 = real(ifft2(fft2(bstr).*filt));
sigmaPSD = sigma^2*abs(filt).^2;
BMopts.profile = 'default';
u4 = GBM3D(u3,sigmaPSD,BMopts);


NL = 20;
lambdaTest = linspace(-3,0,NL);
lambdaTest = 10.^lambdaTest;
ee = zeros(NL,1);
for i = 1:NL
    lambdaTmp = lambdaTest(i);
    filt = conj(hhat).*conj(ghat)./(hhat.*conj(hhat).*conj(ghat).*ghat/K^2 + lambdaTmp*V);
    uTmp = real(ifft2(fft2(bstr).*filt));
    ee(i) = myPSNR(x,uTmp,1);
end

[~,best] = max(ee);
best = lambdaTest(best);





%%
dataStr = imresize(b,K,'nearest');
figure(217);colormap(gray);tiledlayout(2,2);
t1 = nexttile;
imagesc(imresize(b,K,'nearest'),[0 1]);
title(sprintf('blurry im: PSNR = %g',myPSNR(x,dataStr,1)));
t2 = nexttile;
imagesc(x);
t3 = nexttile;
imagesc(out2.U,[0 1]);
title(sprintf('lambda + PSF SURE'));
t4 = nexttile;
imagesc(U,[0 1]);
title(sprintf('lambda only'))
linkaxes([t1 t2]);



figure(128);hold off;
semilogx(lambdaTest,ee);hold on;
plot(real(out.lambdas(end))*ones(2,1),[26 34]);

fprintf('SURE recovered lambda at standard res: %g\n',out.lambdas(end));
fprintf('Ideal lambda at super res: %g\n',best);
fprintf('ratio: %g\n',out.lambdas(end)/best);