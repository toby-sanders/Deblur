clear;
SNR = 20;
omega = 2;
order = 2;
levels = 1;
angle = 0;
% type1 = 'gaus';
type = 'gaus';
rng(2021);
channel = 2;
addBlur = false;
pad = 20;
SNR = 30;
% xx = 3101:3800;
% yy = 3201:3900;
% xx = 2001:2500;
% yy = 3601:4100;
xx = 7501:8600;
yy = 1:900;
% yy = 5601:6050;
% xx = 1901:2400;
% yy = 5001:5300;
% xx = 10241:10390;
% xx = 501:1100;
% yy = 6101:6650;
% xx = 7169:7169+255;
% yy = 6401:6400+256;
% xx = 41:1050;
% yy = 451:2500;


% get image and add noise
% path = 'C:\Users\toby.sanders\Dropbox\TobySharedMATLAB\Surdex';
% I = im2double(imread([path,filesep,'surdex1.tif']));
% I = I0(yy,xx,channel);
% path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\jupiter\';
% fname = 'jupiterComet.tif';
path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\';
fname = 'bigradient_conv.tif';
I = (im2double((imread([path,fname]))));
% I = rgb2gray(im2double(imread('saturn.png')));
% I = I(yy,xx);
I = padEdgesColorIm(I,pad);
[d1,d2] = size(I);
[I,sigma] = add_Wnoise(I,SNR);
b = I;

V = my_Fourier_filters(order,levels,d1,d2,1);
% V = 200./abs(fft2(b)).^2;
Fb = fft2(b);

% sigma = determineNoise1D(b,10);


opts.sigma = sigma;
opts.V = V;
opts.iter = 250;
opts.tol = 1e-5;


% compare PSF widths found with the fixed point iteration
omega0 = 1;
opts.tol = 1e-7;
opts.lambda = 8.4e2*sigma^2;
opts.gamma = .1;
opts.iter = 600;
% out1 = SURE_FP_PSF2D(Fb,omega0,opts);
out2 = SURE_FP_PSF3D(Fb,omega0,opts);

% evaluate improved BM3D solutions with recovered parameters
omegaEX = out2.omegasX(end);
omegaEY = out2.omegasY(end);
thetaE = out2.thetas(end);
[h,hhat] = makeGausPSF([d1,d2],omegaEX,omegaEY,thetaE);
% [h2,hhat2] = makeGausPSF([d1,d2],out1.omegasX(end),out1.omegasY(end),0);
filt1 = conj(hhat)./(abs(hhat).^2 + out2.lambdas(end).*opts.V);
% filt2 = conj(hhat2)./(abs(hhat2).^2 + out2.lambdas(end).*opts.V);
rec1 = real(ifft2(Fb.*filt1));
% rec10 = real(ifft2(Fb.*filt2));
BMopts.profile = 'fast';
rec2 = GBM3D_deconv(b,hhat,sigma,BMopts);

opts.profile = 'default';
sigmaPSD = sigma^2*abs(filt1).^2;
rec5 = GBM3D(rec1,sigmaPSD,opts);




%%
% omGuessX = .1;
% omGuessY = 4;
% [h2,hhat2] = makeGausPSF([d1,d2],omGuessX,omGuessY,0,1);
% filt2 = conj(hhat2)./(abs(hhat2).^2 + out2.lambdas(end).*opts.V);
% rec6 = real(ifft2(Fb.*filt2));


mm1 = 0.0;
mm2 = 1;
gam = 1;
figure(537);tiledlayout(2,3,'tilespacing','compact');
nexttile;plot(out2.thetas);title('theta convergence');
nexttile;hold off;
plot(out2.omegasX);hold on;
plot(out2.omegasY);title('omega convergence (with theta');
legend('omegaX','omegaY');
nexttile;semilogy(out2.lambdas);title('lambda conv');
t1 = nexttile;
imagesc(fftshift(h)/max(h(:)));title('angled PSF recovery')
axis([round(d2/2)-5,round(d2/2)+6,round(d1/2)-5,round(d1/2)+6]);
nexttile;
loglog(out2.UPREs);

figure(538);tiledlayout(2,2,'tilespacing','compact');colormap(gray);
% plot([width, width],[min(SUREall),max(SUREall)],':'); hold off;
% t7 = nexttile;imagesc(b,[mm1 mm2]);title('blurry');
t3 = nexttile;imagesc(abs(I).^gam,[mm1,mm2]);title('original');
t4 = nexttile;imagesc(abs(rec1).^gam,[mm1,mm2]);title('estimated PSF Wiener');
t5 = nexttile;imagesc(abs(rec2).^gam,[mm1,mm2]);title('estimated PSF BM3D')
t8 = nexttile;imagesc(abs(rec5).^gam,[mm1 mm2]);title('Wiener + BM3D');
% t9 = nexttile;imagesc(abs(rec6).^gam,[mm1 mm2]);title('non-angle Wiener rec');
linkaxes([t3 t4 t5 t8]);

