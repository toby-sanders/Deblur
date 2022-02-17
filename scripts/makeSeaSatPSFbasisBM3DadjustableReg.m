% testing deconvolution of the seasat data with the SURE approach using the
% linear combination of multiple psfs

clear;
rng(2022);

% load in data and set telescope parameters
fname =  'C:\Users\toby.sanders\Desktop\Seasat Data Set\SAT10967_0001.hd5';
blurred = h5read(fname,'/SensorPC/Data');
background = h5read(fname,'/SensorPC/Background');
yy = 126:425;
xx = 76:375;
blurred = blurred(yy,xx,:);
background = background(yy,xx,:);
slice = 65; % slice number to start in movie
waveLength = 1000;
nRads = 80;
teleDiam = 3.5;
obscureDiam = .7;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% user parameters this code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
blockSize = 1; % number of blocks in frame (only works for 1 now)
sigmaFudge = 1.0; % fudge factor for noise level estimator
BMfudge = 2.0; % fudge factor for BM3D deblur code
telePSF = 32; % number of random PSFs generated from telescope model
Nf = 32; % total number of PSFs to generate (remainder will be Gaussian)
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
sigma = sigmaFudge*determineNoise1D(b,20);

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

%% after estimating PSF, run more advanced deconv. algorithms

%%
BMopts.profile = 'default';
% BM3D
U = out2.U;
UB = GBM3D_deconv(b,out2.hhat,sigma*BMfudge,BMopts);
UB3 = GBM3D_deconv(b,out2.hhat,sigma,BMopts);
BMopts.alpha = out2.lambdas(end)/10;
UB4 = GBM3D_deconv(b,out2.hhat,sigma,BMopts);
%%
BMopts.alpha = 27*sigma^2;
[UB2,outBM] = GBM3D_deconv2(b,out2.hhat,sigma,BMopts);




% display 
fprintf('SURE final value: %g\n',out2.UPREs(end))
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


figure(909);tiledlayout(2,3);colormap(gray);

t3 = nexttile;imagesc(max(b,0).^gam,[mm1 mm2].^gam);
title(sprintf('blurry/noisy'));
t1 = nexttile;imagesc(max(outBM.U1,0).^gam,[mm1 mm2].^gam);
title('1 step bm3d deblur');
t2 = nexttile;imagesc(UB2.^gam,[mm1 mm2].^gam);
title('BM3D deblur, relaxed');
t4 = nexttile;
imagesc(UB4.^gam,[mm1 mm2].^gam);
title('BM3D deblur, SURE lambda used')
% imagesc(recMF.^gam,[0 mm3.^gam]);colorbar;
% title('RL MFBD')
t5 = nexttile;
imagesc(UB3.^gam,[mm1 mm2].^gam);colorbar;
title('BM3D deblur')
t6 = nexttile;
imagesc(UB.^gam,[mm1 mm2].^gam);colorbar;
title('BM3D deblur with fudge')
% imagesc(UB.^gam,[mm1 mm2]);
% title('BM3D deblur');
linkaxes([t1 t2 t3 t4 t5 t6]);


figure(903);
subplot(1,2,1);
imagesc(abs(real(fftshift(hR))).^gamP);colorbar;
axis([d2/2-15,d2/2+15,d1/2-15,d1/2+15]);
title('SURE recovered psf');
subplot(1,2,2);
imagesc(real(hR0).^gamP);colorbar;
axis([d2/2-15,d2/2+15,d1/2-15,d1/2+15]);
title('MFBD recovered psf');




%%
figure(1009);colormap(gray);
tiledlayout(2,2);
t1 = nexttile;
imagesc(max(outBM.recWie,0).^gam,[mm1,mm2].^gam);
title('BM first Wiener rec')
t2 = nexttile;
imagesc(max(outBM.recWie2,0).^gam,[mm1,mm2].^gam);
title('BM second Wiener rec, relaxed');
t3 = nexttile;
imagesc(max(outBM.U1,0).^gam,[mm1,mm2].^gam);
title('BM first step');
t4 = nexttile;
imagesc(UB2.^gam,[mm1,mm2].^gam);
title('BM second step, relaxed');
linkaxes([t1 t2 t3 t4]);

