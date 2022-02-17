clear;
K = 2;
order = 2;
fudgeFactor = 1.0;
pad = 10;

% xx = 8601:9100;
% yy = 3301:3900;

xx = 600:1200;
yy = 6001:6800;

% xx = 7501:8600;
% yy = 1:600;
% xx = 7257:7512;
% yy = 5941:5940+256;
% yy = 301:700;
% xx = 901:1500;
% xx = 1951:2400;
% yy = 2401:3300;
% yy = 4800:5300;
% xx = 7950:8350;


% get image and add noise
% path = 'C:\Users\toby.sanders\Dropbox\TobySharedMATLAB\Surdex';
% path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\NGA';
path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages';
I0 = (im2double(imread([path,filesep,'surdex1.tif'])));
I0 = rgb2gray(I0(yy,xx,1:3));
% I0 = padEdgesColorIm(I0,30);
% I = I0;
% I = I0(yy,xx);
% I0 = im2double(imread([path,filesep,'tanks.tif']));
I = I0;
% I = I0(yy,xx,1);







I = padEdgesColorIm(I,pad);
[d1,d2] = size(I);
sigma = determineNoise1D(I,10);
sigma = sigma*fudgeFactor;
Ihat = fft2(I);
V = my_Fourier_filters(order,1,d1,d2,1);

opts.PSF = 'gaus';
opts.gamma = .10;
opts.iter = 120;
opts.tol = 1e-6;
opts.V = V;
opts.lambda = 8.4e2*sigma^2;
opts.sigma = sigma;
out = SURE_FP_PSF3D(Ihat,1,opts);
h0 = real(ifft2(out.hhat));


omegaX = out.omegasX(end);
omegaY = out.omegasY(end);
theta = out.thetas(end);

% original blur
% [h,hhat] = makeGausPSF(d0,omega);
% blur kernel for downsampling
g = zeros(d1*K,d2*K);
g([1:K],[1:K]) = 1/K^2;
g = fraccircshift(g,[-K/2 + 1/2, -K/2 + 1/2]);
ghat = fft2(g);

BMopts.profile = 'default';
u5 = GBM3D_deconv(I,out.hhat,sigma,BMopts);

% simulate data
b = I;
bstr = zeros(d1*K,d2*K);
bstr(1:K:end,1:K:end) = b;
[h,hhat] = makeGausPSF([d1*K,d2*K],omegaX,omegaY,theta,1/K);

V2 = my_Fourier_filters(order,1,d1*K,d2*K,1);
lambda = 8.4e2*sigma^2; % reg. parameter
% lambda = out.lambdas(end);
% u3 is done by approximating matrix inverse with diagonal
filt = conj(hhat).*conj(ghat)./(hhat.*conj(hhat).*conj(ghat).*ghat/K^2 + lambda*V2);
u3 = real(ifft2(fft2(bstr).*filt));


sigmaPSD = sigma^2*abs(filt).^2;
u4 = GBM3D(u3,sigmaPSD,BMopts);



A = getSuperResDeblurOpers(d1*K,d2*K,K,hhat,ghat);
Reg = getGBM3D;
popts.sigma0 = sigma*0.75;% 1/255;
popts.iter = 25;
popts.tol = 1e-5;
popts.init = u3;
[p3rec,p3out] = PnP3_prox(A,Reg,b(:),[d1*K,d2*K,1],popts);



%%
mm1 = 0.3;
mm2 = 1;
figure(22);colormap(gray);
tiledlayout(2,3);
t1 = nexttile;
imagesc(imresize(b,K,'nearest'),[mm1 mm2]);
title('original');
t5 = nexttile;
imagesc(imresize(u5,K,'nearest'),[mm1,mm2]);
title('standard BM3D deblur');
t2 = nexttile;
imagesc(imresize(out.U,K,'nearest'),[mm1 mm2]);
title('Wiener deblur');
t4 = nexttile;
imagesc(u4,[mm1 mm2]);
title('super res. Wiener + BM3D');
t6 = nexttile;
imagesc(p3rec,[mm1 mm2]);
title('P3 super res');
t3 = nexttile;
imagesc(u3,[mm1 mm2]);
title('super res. Wiener deblur')
linkaxes([t1 t2 t3 t4 t5 t6])


figure(213);colormap(gray);
tiledlayout(2,3);
t1 = nexttile;
imagesc(imrotate(imresize(b,K,'nearest'),-90),[mm1 mm2]);
title('original');
t5 = nexttile;
imagesc(imrotate(imresize(u5,K,'nearest'),-90),[mm1,mm2]);
title('standard BM3D deblur');
t2 = nexttile;
imagesc(imrotate(imresize(out.U,K,'nearest'),-90),[mm1 mm2]);
title('Wiener deblur');
t4 = nexttile;
imagesc(imrotate(u4,-90),[mm1 mm2]);
title('super res. Wiener + BM3D')
t6 = nexttile;
imagesc(imrotate(p3rec,-90),[mm1,mm2]);
title('P3 super res')
t3 = nexttile;
imagesc(imrotate(u3,-90),[mm1 mm2]);
title('super res. Wiener deblur')
linkaxes([t1 t2 t3 t4 t5 t6]);


figure(214);
subplot(2,3,1);hold off;
plot(out.omegasX);hold on;
plot(out.omegasY);hold off;
title('omegas')
subplot(2,3,2);
% semilogy(out.UPREs);title('SUREs');
imagesc(-d2:d2,-d1:d1,fftshift(h));
axis([-10 10 -10 10]);
title('PSF at new res.')
subplot(2,3,3);
semilogy(out.lambdas);title('lambdas')
subplot(2,3,4);
plot(out.thetas);
title('thetas');
subplot(2,3,5);
loglog(out.UPREs);
title('SURE values');
ylabel('iteration');
subplot(2,3,6);
imagesc(-d2:d2,-d1:d1,fftshift(h0));
axis([-10 10 -10 10]);
title('PSF out from SURE');








