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
pad = 10;
K = 2;


Nf = 49;
minSD = .1;
maxSD = 2.5;

% xx = 1001:4000;
% yy = 2501:5500;
xx = 7001:7800;
yy = 4501:6500;


xx = 600:2200;
yy = 6001:7800;

% xx = 600:950;
% yy = 6001:6400;


% get image and add noise
path = 'C:\Users\toby.sanders\Dropbox\TobySharedMATLAB\Surdex';
I0 = im2double(imread([path,filesep,'surdex1.tif']));
IC = I0(yy,xx,1:3);
I = rgb2gray(I0(yy,xx,1:3));
% path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\';
% path = 'C:\Users\toby.sanders\Dropbox\archives\data\dataImages\';
% I = rgb2gray(im2double((imread([path,'MOR4.tif']))));
% I = I(201:820,251:1500);
[d1,d2] = size(I);
[R,~,Bover] = getDeblur_distribution(d1,d2,d0);





% initialize variables in loop
M = size(R,3);
N = size(R,2);
rec = zeros(d1,d2,3);
recB = zeros(d1,d2,3);
recB2 = zeros(d1,d2,3);
recS = zeros(d1*K,d2*K,3);
recSW = recS;
normalizer = zeros(d1,d2);
normalizerS = zeros(d1*K,d2*K);
sigmaA = zeros(M,N);
alphaAll = zeros(Nf,M,N);
out = cell(M,N);

zetasX = (rand(1,Nf)+minSD)*maxSD;
zetasY = (rand(1,Nf)+minSD)*maxSD;
phis = (rand(1,Nf)-1/2)*90;





% loop over each image patch

for i = 1:M
    for j = 1:N

        % get local image patch
        indX = R(3,j,i):R(4,j,i);
        indY = R(1,j,i):R(2,j,i);
        b = I(indY,indX);
        b = padEdgesColorIm(b,pad);
        bC = IC(indY,indX,:);
        bC = padEdgesColorIm(bC,pad);

        % Fourier transform of image and determine noise
        [m,n] = size(b);
        V = my_Fourier_filters(order,levels,m,n,1);
        Fb = fft2(b);
        FbC = fft2(bC);
        sigma = determineNoise1D(b,10);
        

        f = zeros(m,n,Nf);
        fhat = zeros(m,n,Nf);
        for cc = 1:Nf
            [f(:,:,cc),fhat(:,:,cc)] = makeGausPSF([m,n],zetasX(cc),zetasY(cc),phis(cc),1);
        end
        
        
        fS = zeros(m*K,n*K,Nf);
        fShat = fS;
        for cc = 1:Nf
            [fS(:,:,cc),fShat(:,:,cc)] = makeGausPSF([m*K,n*K],zetasX(cc),zetasY(cc),phis(cc),1/K);
        end
        % set options and run SURE fixed point algorithm
        opts.sigma = sigma;
        opts.V = V;
        opts.iter = 120;
        opts.tol = 1e-5;
        % opts.gamma = 1/10;
        % opts.PSF = 'gaus';  % PSF model
        opts.lambda = 8.4e2*sigma^2; % initial lambda guess
        out{i,j} = SURE_FP_PSF_multiPSF(Fb,fhat,opts);
        % out{i,j} = SURE_FP_PSF3D(Fb,omega0,opts);

        % save output parameter values
        alphaAll(:,i,j) = out{i,j}.alphas(:,end);
        sigmaA(i,j) = sigma;
        hhat = out{i,j}.hhat;
        lambda = out{i,j}.lambdas(end);

        % evaluate BM3D solutions with recovered parameters
        BMopts.profile = 'default';
        tmp = GBM3D_deconv(b,out{i,j}.hhat,sigma,BMopts);
        
        % denoise SURE solution with BM3D
        filtRI = conj(hhat)./(abs(hhat).^2 + lambda*opts.V);
        sigmaPSD = sigma^2*abs(filtRI).^2;
        BMopts.profile = 'default';
        tmp = zeros(m,n,3);
        tmp2 = zeros(m,n,3);
        for cc = 1:3
            tmp(:,:,cc) = real(ifft2(FbC(:,:,cc).*filtRI));
            tmp2(:,:,cc) = GBM3D(tmp(:,:,cc),sigmaPSD,BMopts);
        end



        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%  Run Super resolution
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % initialize necessary upsampling variables
        V2 = my_Fourier_filters(order,1,m*K,n*K,1);
        fFinal = zeros(m*K,n*K);
        for cc = 1:Nf
            fFinal = fFinal + alphaAll(cc,i,j)*fShat(:,:,cc);
        end
        % [~,fhat] = makeGausPSF([m*K,n*K],omegaEX,omegaEY,thetaE,1/K);
        bstr = zeros(m*K,n*K,3);
        bstr(1:K:end,1:K:end,:) = bC;


        % blur kernel for downsampling
        g = zeros(m*K,n*K);
        g([1:K],[1:K]) = 1/K^2;
        g = fraccircshift(g,[-K/2 + 1/2, -K/2 + 1/2]);
        ghat = fft2(g);

        lambda = 8.4e2*sigma^2; % empirical lambda

        % Empirical Wiener filter solution
        filtRIS = conj(fFinal).*conj(ghat)./(fFinal.*conj(fFinal).*conj(ghat).*ghat/K^2 + lambda*V2);
        sigmaPSD = sigma^2*abs(filtRIS).^2;
        tmp3 = zeros(m*K,n*K,3);
        tmp4 = zeros(m*K,n*K,3);
        for cc = 1:3
            tmp3(:,:,cc) = real(ifft2(fft2(bstr(:,:,cc)).*filtRIS));
            tmp4(:,:,cc) = GBM3D(tmp3(:,:,cc),sigmaPSD,BMopts); % clean up with BM3D
        end

        % delete padding on all solutions
        tmp = tmp(1:end-pad,1:end-pad,:);
        tmp2 = tmp2(1:end-pad,1:end-pad,:);
        tmp3 = tmp3(1:end-pad*K,1:end-pad*K,:);
        tmp4 = tmp4(1:end-pad*K,1:end-pad*K,:);
        out{i,j}.U = out{i,j}.U(1:end-pad,1:end-pad,:);
        [m,n,~] = size(tmp);

        % put restored local images into global image variables
        w = makeDeblurInterpWindow(m,n,Bover);
        normalizer(indY,indX) = normalizer(indY,indX) + w;
        rec(indY,indX,:)  = rec(indY,indX,:) + out{i,j}.U.*w;
        recB(indY,indX,:)  = recB(indY,indX,:) + tmp.*w;
        recB2(indY,indX,:)  = recB2(indY,indX,:) + tmp2.*w;
        
        % repeat for super resolution images
        wS = makeDeblurInterpWindow(m*K,n*K,Bover*K);
        indYS = indY(1)*2-1:indY(end)*2;
        indXS = indX(1)*2-1:indX(end)*2;
        normalizerS(indYS,indXS) = normalizerS(indYS,indXS) + wS;
        recSW(indYS,indXS,:) = recSW(indYS,indXS,:) + tmp3.*wS;
        recS(indYS,indXS,:) = recS(indYS,indXS,:) + tmp4.*wS;


    end
end


% normalize to correct for overlapping regions
rec = rec./normalizer;
recB = recB./normalizer;
recB2 = recB2./normalizer;
recSW = recSW./normalizerS;
recS = recS./normalizerS;



%% display


% make the tiling image just for display
T = zeros(d1,d2);
for i = 1:size(R,2)
    for j = 1:size(R,3)
        indX = R(3,i,j):R(4,i,j);
        indY = R(1,i,j):R(2,i,j);
        T(indY,indX) = T(indY,indX) + 1;
    end
end

% make an "average" PSF just for display
d00 = 7;
alphaMean = sum(sum(alphaAll,2),3)/M/N;
hF = zeros(d00);
hFS = zeros(d00*K-1);
for i = 1:Nf
    f = makeGausPSF(d00,zetasX(i),zetasY(i),phis(i),1);
    fS = makeGausPSF(d00*K-1,zetasX(i),zetasY(i),phis(i),1/K);
    hF = hF + alphaMean(i)*f;
    hFS = hFS + alphaMean(i)*fS;
end
mmax = 1.2;
mmin = 0.5;
figure(572);tiledlayout(3,3);colormap(jet);
% t1 = nexttile;
% imagesc(omegaXA,[mmin mmax]);colorbar;
% title('omega, X')
% t2 = nexttile;
% imagesc(omegaYA,[mmin mmax]);colorbar;
% title('omega, Y')
% t3 = nexttile;
% imagesc(thetaA);
% colorbar;title('angles, theta');
t4 = nexttile;
imagesc(sigmaA);title('noise estimates, sigma')
colorbar;
nexttile;
imagesc(T);title('image tiling/overlap');colorbar;
nexttile;
imagesc(fftshift(hF));
title('average PSF');
nexttile;
imagesc(fftshift(hFS));
title('average super res PSF');
nexttile;
imshowpair(T,I);


mm1 = .4;
mm2 = 1;
figure(574);colormap(gray);
tiledlayout(2,2);
% t4 = nexttile;
% imagesc(rec,[0.4 1]);colorbar;title('recovered');
% t6 = nexttile;
% imagesc(recB,[0.4, 1]);colorbar;
% title('BM3D Deblur');
t5 = nexttile;
imagesc(imresize(IC,K,'nearest'),[0.4 1]);colorbar;
title('original');
t7 = nexttile;
imagesc(imresize(recB2,K,'nearest'),[0.4 1]);
title('BM3D correction');
t6 = nexttile;
imagesc(recS,[mm1,mm2]);title('super resolution BM3D');
t8 = nexttile;
imagesc(recSW,[mm1,mm2]);title('super resolution Wiener');
linkaxes([t5 t7 t6 t8])

mm1 = .2;
figure(210);colormap(gray);
tiledlayout(3,2,'tilespacing','none');
tt1 = nexttile;imshow(imresize((IC-mm1)/(1-mm1),K,'nearest'));title('original imagery');
tt2 = nexttile;imshow((recS-mm1)/(1-mm1));title('Super resolution deconvolved imagery')
tt11 = nexttile;imshow(imresize((IC-mm1)/(1-mm1),K,'nearest'));
tt22 = nexttile;imshow((recS-mm1)/(1-mm1));
tt111 = nexttile;imshow(imresize((IC-mm1)/(1-mm1),K,'nearest'));
tt222= nexttile;imshow((recS-mm1)/(1-mm1));
linkaxes([tt1 tt2]);
linkaxes([ tt11 tt22]);
linkaxes([ tt111 tt222]);
set([tt1,tt2 tt22 tt11 tt111 tt222],'Xtick',[],'Ytick',[]);


for i = 1:49
    figure(12);
    f = makeGausPSF(d00,zetasX(i),zetasY(i),phis(i),1);
    subplot(1,2,1);imagesc(squeeze(alphaAll(i,:,:)));title('coefficient values');
    colorbar;
    subplot(1,2,2);imagesc(fftshift(f));title(sprintf('PSF number %i',i));
    
    colorbar;
    pause;
end
