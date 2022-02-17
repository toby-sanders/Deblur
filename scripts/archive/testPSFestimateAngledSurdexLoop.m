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
d0 = 256;
% xx = 1001:4000;
% yy = 2501:5500;
xx = 7001:7800;
yy = 4501:6500;


xx = 600:2200;
yy = 6001:7800;




% get image and add noise
path = 'C:\Users\toby.sanders\Dropbox\TobySharedMATLAB\Surdex';
I0 = im2double(imread([path,filesep,'surdex1.tif']));
I0 = rgb2gray(I0(:,:,1:3));
I = I0(yy,xx,1);
% path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\';
% path = 'C:\Users\toby.sanders\Dropbox\archives\data\dataImages\';
% I = rgb2gray(im2double((imread([path,'MOR4.tif']))));
% I = I(201:820,251:1500);
[d1,d2] = size(I);
R = getDeblur_distribution(d1,d2);


% initialize variables in loop
M = size(R,3);
N = size(R,2);
rec = zeros(d1,d2);
recB = zeros(d1,d2);
recB2 = zeros(d1,d2);
normalizer = zeros(d1,d2);
omegaXA = zeros(M,N);
omegaYA = omegaXA;
thetaA = omegaXA;
sigmaA = omegaXA;
out = cell(M,N);
% loop over each image patch

pad = 10;
for i = 1:M
    for j = 1:N

        % get local image patch
        indX = R(3,j,i):R(4,j,i);
        indY = R(1,j,i):R(2,j,i);
        b = I(indY,indX);
        b = padEdgesColorIm(b,pad);

        % fourier transform of image and determine noise
        [m,n] = size(b);
        V = my_Fourier_filters(order,levels,m,n,1);
        Fb = fft2(b);
        sigma = determineNoise1D(b,10);
        
        % set options for SURE fixed point algorithm
        opts.sigma = sigma;
        opts.V = V;
        opts.iter = 120;
        opts.tol = 1e-5;
        opts.gamma = 1/10;
        
        % run fixed point SURE
        opts.PSF = 'gaus';  % PSF model
        opts.tol = 1e-7;   % convergence tolerance
        opts.lambda = 8.4e2*sigma^2; % initial lambda guess
        if strcmpi(opts.PSF,'Laplace')
            omega0 = 5; % initial guess for PSF parameter
            opts.gamma = 1;
        else
            omega0 = 1;
            opts.gamma = .1;
        end
        out{i,j} = SURE_FP_PSF3D(Fb,omega0,opts);
        omegaEX = out{i,j}.omegasX(end);
        omegaEY = out{i,j}.omegasY(end);
        thetaE = out{i,j}.thetas(end);
        omegaXA(i,j) = omegaEX;
        omegaYA(i,j) = omegaEY;
        thetaA(i,j) = thetaE;
        sigmaA(i,j) = sigma;
        hhat = out{i,j}.hhat;
        lambda = out{i,j}.lambdas(end);

        % evaluate improved BM3D solutions with recovered parameters
        BMopts.profile = 'default';
        tmp = GBM3D_deconv(b,out{i,j}.hhat,sigma,BMopts);

        filtRI = conj(hhat)./(abs(hhat).^2 + lambda*opts.V);
        sigmaPSD = sigma^2*abs(filtRI).^2;
        tmp2 = GBM3D(out{i,j}.U,sigmaPSD,BMopts);


        tmp = tmp(1:end-pad,1:end-pad,:);
        tmp2 = tmp2(1:end-pad,1:end-pad,:);
        out{i,j}.U = out{i,j}.U(1:end-pad,1:end-pad,:);
        [m,n] = size(tmp);

        % add restored local images into global image variables
        w = makeDeblurInterpWindow(m,n,16);
        normalizer(indY,indX) = normalizer(indY,indX) + w;
        rec(indY,indX)  = rec(indY,indX) + out{i,j}.U.*w;
        recB(indY,indX)  = recB(indY,indX) + tmp.*w;
        recB2(indY,indX)  = recB2(indY,indX) + tmp2.*w;
    end
end
% normalize to correct for overlapping regions
rec = rec./normalizer;
recB = recB./normalizer;
recB2 = recB2./normalizer;

%%


T = zeros(d1,d2);
for i = 1:size(R,2)
    for j = 1:size(R,3)
        indX = R(3,i,j):R(4,i,j);
        indY = R(1,i,j):R(2,i,j);
        T(indY,indX) = T(indY,indX) + 1;
    end
end


mmax = 1.2;
mmin = 0.5;
figure(572);tiledlayout(2,2);colormap(jet);
t1 = nexttile;
imagesc(omegaXA,[mmin mmax]);colorbar;
title('omega, X')
t2 = nexttile;
imagesc(omegaYA,[mmin mmax]);colorbar;
title('omega, Y')
t3 = nexttile;
imagesc(thetaA);
colorbar;title('angles, theta');
t4 = nexttile;
imagesc(sigmaA);title('noise estimates, sigma')
colorbar;

figure(573);colormap(gray);
tiledlayout(1,2);
% t4 = nexttile;
% imagesc(rec,[0.4 1]);colorbar;title('recovered');
% t6 = nexttile;
% imagesc(recB,[0.4, 1]);colorbar;
% title('BM3D Deblur');
t5 = nexttile;
imagesc(I,[0.4 1]);colorbar;
title('original');
t7 = nexttile;
imagesc(recB2,[0.4 1]);
title('BM3D correction');
linkaxes([t5 t7])

