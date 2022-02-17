% testing fixed point methods for optimizing a PSF and regularization
% parameters using SURE
clear;
SNR = 20;
omega = 0.9; % standard deviation of initial PSF
lambdas = linspace(4,-2,30); % test values for lambda (for comparison only)
lambdas = 10.^(-lambdas);
order = 2;
levels = 1;
rng(2021);


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

% make PSF and blurry/noisy image data
[d1,d2] = size(I);
[g,ghat] = makeGausPSF([d1,d2],omega);
b = real(ifft2(fft2(I).*ghat));
[b,sigma] = add_Wnoise(b,SNR);
V = my_Fourier_filters(order,levels,d1,d2,1);
bhat = fft2(b);

% set options
opts.sigma = sigma;
opts.V = V;
opts.lambdas = lambdas;

% first cheat and find the absolute optimal lambda
out0 = SURE_deblur(bhat,ghat,opts);
SUREall = out0.SUREbest;
lambda = out0.lambdaBest;
opts.lambda = lambda;

% initialize some random PSF variance
omega0 = omega*rand(1)*2;

% evaluate SURE FP over PSF for fixed lambda set optimially
out1 = SURE_FP_PSF(bhat,omega0,opts);

% reset lambda and run FP over both PSF and lambda
opts.lambda = lambda*0.5;
outN = SURE_FP(bhat,omega0,opts);

% finally, with output lambda ensure that PSF remains the same
opts.lambda = outN.lambdas(end);
out2 = SURE_FP_PSF(bhat,omega0,opts);

% brute force search for the optimal PSF, just for comparison
Tomegas = linspace(.5,omega*2,15);
out = cell(numel(Tomegas),1);
SUREall = zeros(numel(Tomegas),1);
for i = 1:numel(Tomegas)
    [h,hhat] = makeGausPSF([d1,d2],Tomegas(i));
    out{i} = SURE_deblur(bhat,hhat,opts);
    SUREall(i) = out{i}.SUREbest;
end
[~,ind] = min(SUREall);
omegaB = Tomegas(ind);


%%
figure(89);colormap(gray);
subplot(2,2,1);hold off;
plot(out1.omegas);
legend('omegas');xlabel('iteration');
title(sprintf('final omega  = %g, lambda = %g',out1.omegas(end),lambda));
hold off;
subplot(2,2,2);
semilogy(Tomegas,SUREall);
title(sprintf('brute force omega search: %g',omegaB));
xlabel('omega');ylabel('SURE');
subplot(2,2,3);hold off;
plot(outN.omegas);hold on;
plot(out2.omegas);title(sprintf('final omega: %g',outN.omegas(end)));
xlabel('iteration');ylabel('omega');
hold off;
legend('variable lambda','fixed lambda');
subplot(2,2,4);plot(real(outN.lambdas));title(sprintf('final lambda: %g',outN.lambdas(end)));

figure(88);colormap(gray);tiledlayout(2,2);
t1 = nexttile;imagesc(out0.rBest,[0 1]);
title(sprintf('Correct PSF solution, PSNR = %g',myPSNR(I,out0.rBest,1)));
t2 = nexttile;;imagesc(out2.U,[0 1]);
title(sprintf('PSF iter only, PSNR = %g',myPSNR(I,out2.U,1)));
t3 = nexttile;imagesc(outN.U,[0 1]);
title(sprintf('full FP solution, PSNR = %g',myPSNR(I,outN.U,1)));
t4 = nexttile;imagesc(b,[0 1]);
title(sprintf('blurry image, PSNR = %g',myPSNR(I,b,1)));
linkaxes([t1 t2 t3 t4]);