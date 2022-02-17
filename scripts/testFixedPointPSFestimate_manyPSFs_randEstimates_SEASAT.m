% testing fixed point methods for optimizing a PSF and regularization
% parameters using SURE
clear;
% rng(2029);
SNR = 35;
Nf = 64;
minSD = .1;
maxSD = 3;
zetasX = (rand(1,Nf)+minSD)*maxSD;
zetasY = (rand(1,Nf)+minSD)*maxSD;
phis = (rand(1,Nf)-1/2)*90;
% zeta = 3;

order = 2;
levels = 1;



% get image and add noise
path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\';
% path = '/home/tobysanders/Dropbox/archives/data/testImages/';
% I = im2double(rgb2gray(imread([path,'lena.png'])));
% I = im2double((imread([path,'house.tif'])));I = I(:,:,1);
% I = im2double(rgb2gray(imread([path,'peppers2.png'])));
% I = im2double((imread([path,'confFlag.tif'])));
I = im2double((imread([path,'SeasatBlurry2.tif'])));
% I = im2double(rgb2gray(imread([path,'PineyMO_Oimage.tif'])));
% I = im2double(rgb2gray(imread([path,'monarch.png'])));
% I = im2double((imread('cameraman.tif')));
% I = phantom(512);


% make PSF and blurry/noisy image data
I = I -14/255;
[d1,d2] = size(I);
b = I;
V = my_Fourier_filters(order,levels,d1,d2,1);
bhat = fft2(b);
sigma = determineNoise1D(b,20);

f = zeros(d1,d2,Nf);
fhat = zeros(d1,d2,Nf);
for i = 1:Nf
    [f(:,:,i),fhat(:,:,i)] = makeGausPSF([d1,d2],zetasX(i),zetasY(i),phis(i),1);
end



% set options
opts.sigma = sigma;
opts.V = V;
opts.tol = 1e-4;
opts.iter = 550;


% reset lambda and run FP over both PSF and lambda
opts.lambda = 8.4e2*sigma^2;
tic;
out2 = SURE_FP_PSF_multiPSF(bhat,fhat,opts);
toc;
hR = ifft2(out2.hhat);

%%
U = out2.U;
BMopts.profile = 'default';
fudgeFactor = 1.5;
UB = GBM3D_deconv(b,out2.hhat,sigma*fudgeFactor,BMopts);


fprintf('SURE final value: %g\n',out2.UPREs(end))
gam = .75;

Niter = numel(out2.alphas(1,:));
figure(901);
subplot(3,3,1);hold off;
for i = 1:Nf
    plot(out2.alphas(i,:));hold on;
    % plot(1:Niter,alphas(i)*ones(1,Niter),':k','linewidth',0.5*i);
end
axis([0 Niter,-.1,1.1])
hold off;

subplot(3,3,2);hold off;
plot(real(out2.lambdas));
title('lambda convergence');


subplot(3,3,4);imagesc(real(fftshift(hR)));colorbar;
axis([d2/2-15,d2/2+15,d1/2-15,d1/2+15]);
title('recovered psf');
subplot(3,3,5);hold off;
semilogy(out2.UPREs);hold off;
title('UPRE values');xlabel('iteration');
subplot(3,3,6);plot(out2.normer);
title('normalization needed');
xlabel('iteration');
subplot(3,3,7);plot(out2.numUsed);

mm1 = 0;
mm2 = .8;
figure(902);tiledlayout(1,3);colormap(gray);
t1 = nexttile;imagesc(max(U,0).^gam,[mm1 mm2]);
title(sprintf('SURE deblurred'));
t3 = nexttile;imagesc(max(b,0).^gam,[mm1 mm2]);
title(sprintf('blurry/noisy'));
t4 = nexttile;
imagesc(UB.^gam,[mm1 mm2]);
linkaxes([t1 t3 t4]);

nnn = floor(sqrt(Nf));
W = 5;
figure(903);tiledlayout(nnn,nnn,'tilespacing','none');
for i = 1:nnn^2
    nexttile;imagesc(fftshift(f(:,:,i)));% colorbar;
    % title(sprintf('PSF %i, weight = %g',i,out2.alphas(i,end)));
    axis([d2/2-W,d2/2+W+1,d1/2-W,d1/2+W+1]);
    set(gca,'Xtick',[],'Ytick',[])
end