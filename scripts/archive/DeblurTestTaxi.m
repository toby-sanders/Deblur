clear;
K = 2;
order = 2;
xx = 501:1000;
yy = 301:800;

% get image and add noise
% path = '/home/tobysanders/Dropbox/archives/data/testImages';
path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages';
I0 = im2double(imread([path,filesep,'taxi.jpg']));
I0 = rgb2gray(I0(:,:,1:3));
I = I0(yy,xx);
[d1,d2] = size(I);
sigma = determineNoise(I,10);
% sigma = .015;
Ihat = fft2(I);
V = my_Fourier_filters(order,1,d1,d2,1);

opts.PSF = 'gaus';
opts.gamma = .2;
opts.iter = 250;
opts.tol = 1e-6;
opts.V = V;
opts.lambda = 8.4e2*sigma^2;
opts.sigma = sigma;
out = SURE_FP_PSF3D(Ihat,1,opts);

omegaX = out.omegasX(end);
omegaY = out.omegasY(end);
theta = out.thetas(end);
[h,hhat] = makeGausPSF([d1,d2],omegaX,omegaY,theta);


BMopts.profile = 'default';
u3 = GBM3D_deconv(I,out.hhat,sigma,BMopts);
%%
mm1 = 0;
mm2 = .7;
figure(21);colormap(gray);
tiledlayout(2,2);
t1 = nexttile;
imagesc(I,[mm1 mm2]);
title('original');
t2 = nexttile;
imagesc(out.U,[mm1 mm2]);
title('Wiener deblur');
t3 = nexttile;
imagesc(u3,[mm1 mm2]);
title('GBM3D Wiener deblur')

linkaxes([t1 t2 t3])

figure(214);
subplot(2,2,1);hold off;
plot(out.omegasX);hold on;
plot(out.omegasY);hold off;
title('omegas')
subplot(2,2,2);
% semilogy(out.UPREs);title('SUREs');
imagesc(-d2/2:d2/2,-d1/2:d1/2,fftshift(h));
axis([-10 10 -10 10])
subplot(2,2,3);
semilogy(out.lambdas);title('lambdas')
subplot(2,2,4);
plot(out.thetas);
title('thetas');







