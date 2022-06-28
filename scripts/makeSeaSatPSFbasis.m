% testing deconvolution of the seasat data with the SURE approach using the
% linear combination of multiple psfs

clear;
rng(2023);

% load in data and set telescope parameters
% fname =  'C:\Users\toby.sanders\Desktop\Seasat Data Set\SAT10967_0001.hd5';
fname = 'C:\Users\toby.sanders\Dropbox (Personal)\archives\data\MFBD\Seasat Data Set\SAT10967_0001.hd5'
blurred = h5read(fname,'/SensorPC/Data');
background = h5read(fname,'/SensorPC/Background');
yy = 126:425;
xx = 76:375;
blurred = blurred(yy,xx,:);
background = background(yy,xx,:);
slice = 50; % slice number to start in movie
waveLength = 1000;
nRads = 80;
teleDiam = 3.5;
obscureDiam = .7;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% user parameters this code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
blockSize = 1; % number of blocks in frame (only works for 1 now)
sigmaFudge = 1.0; % fudge factor for noise level estimator
BMfudge = 1.0; % fudge factor for BM3D deblur code
telePSF = 32; % number of random PSFs generated from telescope model
Nf = 64; % total number of PSFs to generate (remainder will be Gaussian)
minSD = .1; % min stand. dev. for PSFs
maxSD = 3; % max stand. dev. for PSFs
TVorder = 2; % hotv order 
TVlevels = 2; % hotv levels
lambdaMF = 1e-3; % lambda for RL MFBD code




% set all the processing parameters for MFBD code
N1 = size(blurred,1);
N2 = size(blurred,2);
d = size(blurred,1);
 
PhP.N = [N1,N2]; % image size
PhP.samplingFactor = 1;  % no super resolution
PhP.D = teleDiam; % main aperture diameter
PhP.Do = obscureDiam; % aperture obscurration diameter
PhP.lambda = waveLength*1e-9;  % wavelength
PhP.dx = nRads*1e-9;  % rads
PhP.K = blockSize; % number of frames

PrP.superRes = false;  
PrP.disp = false;  % display option
PrP.badPixel = 0;  % bad pixel detection
gammaP = 1;   % gamma power in ratio
PrP.lambda = lambdaMF;   % regularization parameter
PrP.shortenPhase = false;  % reduced phase iterations
PrP.objectKnown = false;  % object known (phase estimation only)
PrP.phaseKnown = false;
PrP.iterations = 50;   % total iterations
PrP.prIterations = 100;  % inner loop phase iterations
tolFlag = 0;   % flag to use tolerance     
PrP.tol = 1e-5;   % convergence tolerance
PrP.acceleration = true;
PrP.automateLambda = false;
PrP.tiltFlag = 0;
PrP.order = TVorder;
PrP.levels = TVlevels;
PrP.Mask = ones(N1,N2);




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           NEW INITIAL PHASE ESTIMATE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tmpx = linspace(-PhP.dx*PhP.N(1)/2, PhP.dx*PhP.N(1)/2, PhP.N(2)); 
tmpy = linspace(-PhP.dx*PhP.N(2)/2, PhP.dx*PhP.N(2)/2, PhP.N(1));
[x1,x2]   = meshgrid(tmpx,tmpy);
r       = sqrt(x1.^2 + x2.^2); 
NA        = PhP.D/2; 
sigma_psf = (((0.61*PhP.lambda)/NA)/2.634);% 2.634*sigma = (0.61*lambda/NA) 
aperture  = makeAperture(PhP);
initPSF   = fftshift(exp(-(0.5).*(r./sigma_psf).^2)); 
phase0 = phaseRetrieval(initPSF,aperture,zeros(size(aperture.A))); 

d1 = N1;
d2 = N2;
f = zeros(N1,N2,Nf);
fhat = zeros(N1,N2,Nf);

% first PSF is from the MFBD code
PSF = makePSF(aperture.A, phase0, PhP);
f(:,:,1) = PSF.psf;
fhat(:,:,1) = PSF.H;

% remaining PSFs are randomly generated
% these are the random parameters for these PSFs
zetasX = (rand(1,Nf)+minSD)*maxSD;
zetasY = (rand(1,Nf)+minSD)*maxSD;
phis = (rand(1,Nf)-1/2)*90;

% run phase retrivials to get PSFs matching the MFBD model
for i = 2:telePSF
    [h,hhat] = makeGausPSF([d1,d2],zetasX(i),zetasY(i),phis(i),1);
    phase = phaseRetrieval(h,aperture,zeros(size(aperture.A))); 
    PSF = makePSF(aperture.A, phase, PhP);
    f(:,:,i) = PSF.psf;
    fhat(:,:,i) = PSF.H;
end

% remaining PSFs are just Gaussians
for i = telePSF+1:Nf
    [f(:,:,i),fhat(:,:,i)] = makeGausPSF([d1,d2],zetasX(i),zetasY(i),phis(i),1);
end

% run MFBD code
[out,PSF] = MFBD(blurred(:,:,slice:slice+blockSize-1),phase0,background,...
    aperture,PhP,PrP);
recMF = out.objectEstimate;
hR0 = PSF;

%%

% get image
% path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\';
% I = im2double((imread([path,'SeasatBlurry.png'])));
I = double(blurred(:,:,slice))-background;
I = I/max(I(:));
[d1,d2] = size(I);

b = I;
V = my_Fourier_filters(2,1,d1,d2,1);
bhat = fft2(b);
% sigma = sigmaFudge*determineNoise1D(b,20);
% estimate noise on high intesity region of image, since noise is poisson
sigma = determineNoise1D(b(75:160,121:210),20);

% set SURE options and run
opts.sigma = sigma;
opts.V = V;
opts.tol = 1e-4;
opts.iter = 250;
opts.lambda = 8.4e2*sigma^2;
tic;
out2 = SURE_FP_PSF_multiPSF(bhat,fhat,opts);
toc;
hR = ifft2(out2.hhat);


% run the phase retrival on the recovered PSF and form that PSF, which will
% match the telescope optics
phaseF = phaseRetrieval(hR,aperture,zeros(size(aperture.A))); 
hRP = makePSF(aperture.A, phaseF, PhP);
hRPhat = hRP.H;
hRP = hRP.psf;


%% after estimating PSF, run more advanced deconv. algorithms

% BM3D
U = out2.U;
BMopts.profile = 'default';
UB = GBM3D_deconv(b,out2.hhat,sigma*BMfudge,BMopts);
BMopts.alpha = out2.lambdas(end)/10;
UB2 = GBM3D_deconv(b,out2.hhat,sigma,BMopts);

% MHOTV
hopts.mode = 'deconv';
hopts.mu = 10;
hopts.order = TVorder;
hopts.levels = TVlevels;
hopts.nonneg = true;
[UTV,outH] = HOTV3D(hR,ifft2(bhat),[N1,N2],hopts);
[UTV2,outH2] = HOTV3D(hRP,ifft2(bhat),[N1,N2],hopts);

Usharp = LTsharpen(UTV,0.5,1/255);

%% display 
fprintf('SURE final value: %g\n',out2.UPREs(end))
fprintf('sigma values: %g\n',sigma);
gam = 0.75;
gamP = 0.75;
mm1 = 0;
mm2 = 0.8;
mm3 = 2.2e4;


Niter = numel(out2.alphas(1,:));
figure(901);
subplot(3,3,1);hold off;
for i = 1:telePSF
    plot(out2.alphas(i,:));hold on;
    % plot(1:Niter,alphas(i)*ones(1,Niter),':k','linewidth',0.5*i);
end
axis([0 Niter,-.1,1.1])
hold off;
title('coefficients for tele PSFS')

subplot(3,3,2);hold off;
for i = telePSF+1:Nf
    plot(out2.alphas(i,:));hold on;
    % plot(1:Niter,alphas(i)*ones(1,Niter),':k','linewidth',0.5*i);
end
axis([0 Niter,-.1,1.1])
hold off;
title('coef. for Gaus PSFs');

subplot(3,3,3);hold off;
plot(real(out2.lambdas));
title('lambda convergence');

subplot(3,3,5);hold off;
semilogy(out2.UPREs);hold off;
title('SURE values');xlabel('iteration');
subplot(3,3,6);plot(out2.normer);
title('normalization needed');
xlabel('iteration');
subplot(3,3,7);plot(out2.numUsed);
title('number of nonzero coef');
xlabel('iteration');
subplot(3,3,8);
plot(out2.alphas(:,end));


figure(909);tiledlayout(3,3);colormap(gray);

t3 = nexttile;imagesc(max(b,0).^gam,[mm1 mm2].^gam);
title(sprintf('blurry/noisy'));
t1 = nexttile;imagesc(max(U,0).^gam,[mm1 mm2].^gam);
title(sprintf('SURE deblurred'));
t2 = nexttile;imagesc(UTV.^gam,[mm1 mm2].^gam);
title('MHOTV deblur');
t4 = nexttile;
imagesc(recMF.^gam,[0 mm3.^gam]);colorbar;
title('RL MFBD')
t5 = nexttile;
imagesc(UB.^gam,[mm1 mm2].^gam);colorbar;
title('BM3D deblur')
t6 = nexttile;
imagesc(UB2.^gam,[mm1 mm2].^gam);colorbar;
title('BM3D deblur, manual alpha reg')
t7 = nexttile;
imagesc(UTV2.^gam,[mm1 mm2].^gam);colorbar;
title('MHOTV deblur with phase PSF')
% imagesc(UB.^gam,[mm1 mm2]);
% title('BM3D deblur');
linkaxes([t1 t2 t3 t4 t5 t6 t7]);

nnn = floor(sqrt(Nf));
W = 5;
figure(903);tiledlayout(nnn,nnn,'tilespacing','none');
for i = 1:nnn^2
    nexttile;imagesc(fftshift(f(:,:,i)));% colorbar;
    % title(sprintf('PSF %i, weight = %g',i,out2.alphas(i,end)));
    axis([d2/2-W,d2/2+W+1,d1/2-W,d1/2+W+1]);
    set(gca,'Xtick',[],'Ytick',[])
end




figure(1009);
subplot(2,2,1);
imagesc(abs(real(fftshift(hR))).^gamP);colorbar;
axis([d2/2-15,d2/2+15,d1/2-15,d1/2+15]);
title('SURE recovered psf');
subplot(2,2,2);
imagesc(real(hR0).^gamP);colorbar;
axis([d2/2-15,d2/2+15,d1/2-15,d1/2+15]);
title('MFBD recovered psf');
subplot(2,2,3);
imagesc(abs(real(fftshift(hRP))).^gamP);colorbar;
axis([d2/2-15,d2/2+15,d1/2-15,d1/2+15]);
title('SURE recovered psf -> phase retrival');

%%
figure(1010);colormap(gray);
tiledlayout(1,3,'tilespacing','compact');
t1 = nexttile;
imagesc(max(b,0).^gam,[mm1 mm2].^gam);
title(sprintf('blurry satellite image'));
t2 = nexttile;
imagesc(max(UTV,0).^gam,[mm1 mm2].^gam);
title(sprintf('deconvolved image'));
t3 = nexttile;
imagesc(abs(real(fftshift(hRP))).^gamP);
axis([d2/2-15,d2/2+15,d1/2-15,d1/2+15]);
title('Estimated PSF');
linkaxes([t1 t2]);
set([t1 t2 t3 ],'Xtick',[],'Ytick',[],'fontsize',16)