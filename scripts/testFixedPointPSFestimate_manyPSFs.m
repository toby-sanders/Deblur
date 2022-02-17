% testing fixed point methods for optimizing a PSF and regularization
% parameters using SURE
clear;
rng(2025);
SNR = 35;
Nf = 4;
minSD = .1;
maxSD = 3;
omegasX = (rand(1,Nf)+minSD)*maxSD;% [1,2,3,4]*.5;
omegasY = (rand(1,Nf)+minSD)*maxSD;
thetas = (rand(1,Nf)-1/2)*90;
% zeta = 3;
alphas = rand(Nf,1);% [1 0 0];
alphas = alphas/sum(alphas);
order = 2;
levels = 1;



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
Nf = numel(alphas);
g = zeros(d1,d2,Nf);
ghat = g;
h = zeros(d1,d2);
hhat = h;
for i = 1:Nf
    [g(:,:,i),ghat(:,:,i)] = makeGausPSF([d1,d2],omegasX(i),omegasY(i),thetas(i),1);
    h = h + alphas(i)*g(:,:,i);
    hhat = hhat + alphas(i)*ghat(:,:,i);
end
b = real(ifft2(fft2(I).*hhat));
[b,sigma] = add_Wnoise(b,SNR);
V = my_Fourier_filters(order,levels,d1,d2,1);
bhat = fft2(b);

% set options
opts.sigma = sigma;
opts.V = V;
opts.tol = 1e-7;
opts.iter = 550;


% reset lambda and run FP over both PSF and lambda
opts.lambda = 8.4e2*sigma^2;
[U,out] = SURE_FP_Lambda(bhat,hhat,opts);
% out2 = SURE_FP_PSF2D_twoPSFs(bhat,ghat,fhat,opts);
out2 = SURE_FP_PSF_multiPSF(bhat,ghat,opts);
hR = ifft2(out2.hhat);





%%
Niter = numel(out2.alphas(1,:));
figure(901);
subplot(3,3,1);hold off;
for i = 1:Nf
    plot(out2.alphas(i,:));hold on;
    plot(1:Niter,alphas(i)*ones(1,Niter),':k','linewidth',0.5*i);
end
axis([0 Niter,-.1,1.1])
hold off;

subplot(3,3,2);hold off;
plot(real(out.lambdas));hold on;
plot(real(out2.lambdas));
title('lambda convergence');
legend('lambda only','alpha/beta');

subplot(3,3,3);imagesc(fftshift(h));colorbar;
axis([d2/2-15,d2/2+15,d1/2-15,d1/2+15]);
title('true Psf');
subplot(3,3,4);imagesc(real(fftshift(hR)));colorbar;
axis([d2/2-15,d2/2+15,d1/2-15,d1/2+15]);
title('recovered psf');
subplot(3,3,5);hold off;
semilogy(real(out.UPREs));hold on;
semilogy(out2.UPREs);hold off;
title('UPRE values');xlabel('iteration');
legend('lambda only','alpha/beta');
subplot(3,3,6);plot(out2.normer);
title('normalization needed');
xlabel('iteration');

figure(902);tiledlayout(2,2);colormap(gray);
t1 = nexttile;imagesc(U,[0 1]);
title(sprintf('lambda only FP, PSNR = %g',myPSNR(I,U,1)));
t2 = nexttile;imagesc(out2.U,[0 1]);
title(sprintf('alpha/beta FP, PSNR = %g',myPSNR(I,out2.U,1)));
t3 = nexttile;imagesc(b,[0 1]);
title(sprintf('blurry/noisy, PSNR = %g',myPSNR(I,b,1)));
linkaxes([t1 t2 t3]);


figure(903);tiledlayout(2,2);
for i = 1:4
    nexttile;imagesc(fftshift(g(:,:,i)));colorbar;
    title(sprintf('PSF %i, weight = %g, %g',i,alphas(i),out2.alphas(i,end)));
    axis([d2/2-15,d2/2+15,d1/2-15,d1/2+15]);
end