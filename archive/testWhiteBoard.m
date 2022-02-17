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
% xx = 1101:1450;
% yy = 501:900;
% xx = 501:1000;
% yy = 601:1100;
xx = 101:800;
yy = 301:1000;



% get image and add noise
path = 'C:\Users\toby.sanders\Dropbox\TobySharedMATLAB\Surdex';
% I0 = rgb2gray(im2double(imread(['IMG_1505.jpeg'])));
I0 = rgb2gray(im2double(imread(['IMG_1510.jpeg'])));
I0 = rgb2gray(im2double(imread(['IMG_1748.jpeg'])));





I = I0(yy,xx);
% path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\';
% path = 'C:\Users\toby.sanders\Dropbox\archives\data\dataImages\';
% I = rgb2gray(im2double((imread([path,'MOR4.tif']))));
% I = I(201:820,251:1500);
[d1,d2] = size(I);
b = I;


if addBlur
    [~,hbAdd] = makeGausPSF([d1 d2],3,1,20,1);
    % [~,hbAdd] = makeMotionPSF2D([d1,d2],2,2,0,'gaus');
    b = real(ifft2(fft2(b).*hbAdd));
end


V = my_Fourier_filters(order,levels,d1,d2,1);
% V = 200./abs(fft2(b)).^2;
Fb = fft2(b);

sigma = determineNoise1D(b,10);


opts.sigma = sigma;
opts.V = V;
opts.iter = 250;
opts.tol = 1e-5;


% compare PSF widths found with the fixed point iteration
omega0 = 0.5;
opts.tol = 1e-7;
opts.lambda = 8.4e2*sigma^2;
opts.gamma = .1;
opts.PSF = 'gaus';
out2 = SURE_FP_PSF3D(Fb,omega0,opts);








% evaluate improved BM3D solutions with recovered parameters
omegaEX = out2.omegasX(end);
omegaEY = out2.omegasY(end);
thetaE = out2.thetas(end);
if strcmpi(opts.PSF,'Laplace')
    [~,hhat] = makeLaplacePSFAndDer([d1,d2],omegaEX,omegaEY,thetaE);
    h = real(ifft2(hhat));
else
    [h,hhat] = makeGausPSF([d1,d2],omegaEX,omegaEY,thetaE);
end
filt1 = conj(hhat)./(abs(hhat).^2 + out2.lambdas(end).*opts.V);
rec1 = real(ifft2(Fb.*filt1));
BMopts.profile = 'fast';
rec2 = GBM3D_deconv(b,hhat,sigma,BMopts);
rec3 = LTsharpen(rec2,1.5,2/255);

opts.profile = 'default';
sigmaPSD = sigma^2*abs(filt1).^2;
rec5 = GBM3D(rec1,sigmaPSD,opts);



%%
mm1 = 0.25;
mm2 = 1;
gam = 1;
figure(537);tiledlayout(2,3,'tilespacing','compact');colormap(jet);
nexttile;plot(out2.thetas);title('theta convergence');
nexttile;hold off;
plot(out2.omegasX);hold on;
plot(out2.omegasY);title('omega convergence (with theta');
legend('omegaX','omegaY');
nexttile;semilogy(out2.lambdas);title('lambda conv');
t1 = nexttile;
imagesc(fftshift(h));title('PSF recovery')
axis([round(d2/2)-20,round(d2/2)+20,round(d1/2)-20,round(d1/2)+20])

figure(538);tiledlayout(2,3,'tilespacing','compact');colormap(gray);
% plot([width, width],[min(SUREall),max(SUREall)],':'); hold off;
% t7 = nexttile;imagesc(b,[mm1 mm2]);title('blurry');
t3 = nexttile;imagesc(abs(I).^gam,[mm1,mm2]);title('original');
t4 = nexttile;imagesc(abs(rec1).^gam,[mm1,mm2]);title('estimated PSF Wiener');
t5 = nexttile;imagesc(abs(rec2).^gam,[mm1,mm2]);title('estimated PSF BM3D')
t8 = nexttile;imagesc(abs(rec5).^gam,[mm1 mm2]);title('Wiener + BM3D');
linkaxes([t3 t4 t5 t8]);

