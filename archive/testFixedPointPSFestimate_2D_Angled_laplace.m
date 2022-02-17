% testing fixed point methods for optimizing a PSF and regularization
% parameters using SURE
clear;
SNR = 50;
omegaX = 5; % standard deviation of initial PSF
omegaY = 20;
theta = 0;
order = 2;
levels = 1;
rng(2021);
Ttheta = linspace(-85,85,40);

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

% make PSF and blurry/noisy image data
[d1,d2] = size(I);
[~,ghat] = makeLaplacePSFAndDer([d1,d2],omegaX,omegaY,theta);
g = real(ifft2(ghat));
b = real(ifft2(fft2(I).*ghat));
[b,sigma] = add_Wnoise(b,SNR);
V = my_Fourier_filters(order,levels,d1,d2,1);
bhat = fft2(b);

% set options
opts.sigma = sigma;
opts.V = V;
opts.tol = 1e-4;
opts.Utrue = I;

% run the fixed point iteration over all parameters
opts.iter = 200;
opts.PSF = 'Laplace';
outT = SURE_FP_PSF3D(bhat,(omegaX+omegaY)/2,opts);
Niter = numel(outT.omegasX);
thetaR = outT.thetas(end);
omegaXR = outT.omegasX(end);
omegaYR = outT.omegasY(end);
gopts.theta = thetaR;
[~,hhat] = makeLaplacePSFAndDer([d1,d2],omegaXR,omegaYR,thetaR);
h = real(ifft2(hhat));


% Wiener solution based on true PSF
ghat2 = abs(ghat).^2;
Ubest = real(ifft2(bhat.*conj(ghat)./(ghat2 + outT.lambdas(end)*V)));
Aub = col(fft2(Ubest).*ghat - bhat);
trHiAA = sum(ghat2(:)./(V(:)*outT.lambdas(end) + ghat2(:)));
UPREbest = -d1*d2*sigma^2 + Aub'*Aub/(d1*d2) + 2*sigma^2*trHiAA;



opts.profile = 'default';
UBM = GBM3D_deconv(b,outT.hhat,sigma,opts);


%% display
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
yyaxis left
semilogy(outT.lambdas,'k--','linewidth',2);
ylabel('$\lambda$','interpreter','latex');% ,'rotation',0,'Position',[1 1]);
% set(t5.ylabel,'rotation',0)
yyaxis right
plot((outT.thetas),'k','linewidth',2);hold on;
plot(1:Niter,theta*ones(1,Niter),'k:','linewidth',2);
title('convergence of $\theta$ and $\lambda$','interpreter','latex');
ylabel('$\theta$','interpreter','latex');% ,'rotation',0,'Position',[100 32.5]);
xlabel('iteration');% ylabel('angle in degrees')
legend({'$\lambda$','estimated $\theta$','true $\theta$'},'location','east','interpreter','latex');

nexttile;
semilogy(outT.UPREs);
title('SURE values');
xlabel('iteration');


t2 = nexttile;
imagesc(fftshift(g));title('true PSF');colorbar;
t3 = nexttile;
imagesc(fftshift(h));title('recovered PSF');colorbar;
linkaxes([t3 t2]);
axis([d2/2-20 d2/2 + 20 d1/2-20 d1/2 + 20]);



figure(792);tiledlayout(2,2);colormap(gray);
t4 = nexttile;imagesc(b,[0 1]);title(sprintf('blurry/noisy image, PSNR: %g',myPSNR(I,b,1)));
t5 = nexttile;imagesc(outT.U,[0 1]);title(sprintf('recovered image, PSNR: %g',myPSNR(I,outT.U,1)));
t6 = nexttile;imagesc(Ubest,[0 1]);title(sprintf('ideal recovered, PSNR: %g',myPSNR(I,Ubest,1)));
t7 = nexttile;imagesc(UBM,[0 1]);title(sprintf('BM3D deblur, PSNR: %g',myPSNR(UBM,I,1)));
linkaxes([t4 t5 t6 t7]);