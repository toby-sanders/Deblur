clear;
SNR = 20;
omega = 2;
lambdas = linspace(4,-1,20); 
lambdas = 10.^(-lambdas);
order = 2;
levels = 1;
widthX = 1;
widthY = 4;
angle = 0;
type1 = 'gaus';
type2 = 'gaus';
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

% generate blurry data, etc.
[d1,d2] = size(I);
[g,ghat] = makeMotionPSF2D([d1,d2],widthX,widthY,angle,type1);
b = real(ifft2(fft2(I).*ghat));
[b,sigma] = add_Wnoise(b,SNR);
V = my_Fourier_filters(order,levels,d1,d2,1);
% V = 200./abs(fft2(b)).^2;
Fb = fft2(b);

% set options and test values
opts.sigma = sigma;
opts.V = V;
opts.lambdas = lambdas;
opts.trueU = I;
if strcmp(type2,'tophat')
    omegasX = 1:10;
    omegasY = 1:10;
else
    omegasX = linspace(1,widthX*1.5,15);
    omegasY = linspace(1,widthY*1.5,15);
end
nX = numel(omegasX);
nY = numel(omegasY);


% loop to find PSF widths with SURE and brute force search
out = cell(nY,nX);
SUREall = zeros(nY,nX);
SUREallT = zeros(nY,nX);
for i = 1:nY
    for j = 1:nX
        [h,hhat] = makeMotionPSF2D([d1,d2],omegasX(j),omegasY(i),angle,type2);
        out{i,j} = SURE_deblur(Fb,hhat,opts);
        SUREall(i,j) = real(out{i,j}.SUREbest);%  + norm(I(:))^2;
       SUREallT(i,j) = min(out{i,j}.SURET);
    end
end

% find minimum SURE result and widths
[~,ind] = min(SUREall(:));
[indY,indX] = ind2sub([nX,nY],ind);
widthEX = omegasX(indX);
widthEY = omegasY(indY);
lambdaE = out{ind}.lambdaBest;


% compare PSF widths found with the fixed point iteration
omega0 = (widthEX + widthEY)/sqrt(3)/2;
out2 = SURE_FP2(Fb,omega0,opts);

% now run all of the debluring with the recovered parameters
[h,hhat] = makeMotionPSF2D([d1,d2],widthEX,widthEY,angle,type2);
filt1 = hhat./(abs(hhat).^2 + lambdaE.*opts.V);
rec1 = real(ifft2(Fb.*filt1));
BMopts.profile = 'fast';
rec2 = GBM3D_deconv(b,hhat,sigma,BMopts);
rec3 = LTsharpen(rec2,1.5,2/255);


[f,fhat] = makeMotionPSF2D([d1,d2],widthX,widthY,angle,type2);
filt2 = fhat./(abs(fhat).^2 + lambdaE.*opts.V);
rec4 = real(ifft2(Fb.*filt2));

opts.profile = 'default';
sigmaPSD = sigma^2*abs(filt1).^2;
rec5 = GBM3D(rec1,sigmaPSD,opts);



%%
mm1 = 0;
mm2 = 1;
Wresponse = real(filt1.*ghat);
figure(537);tiledlayout(2,2,'tilespacing','compact');colormap(jet);
t2 = nexttile;imagesc(fftshift(Wresponse));
title('response function');colorbar;
t1 = nexttile;hold off;
imagesc(omegasX,omegasY,log(SUREall));% hold on;
colorbar;xlabel('omegaX');ylabel('omegaY');title('SURE values');
t3 = nexttile;hold off;
plot(out2.omegasX);hold on;
plot(out2.omegasY);
plot(1:numel(out2.omegasX),widthEX*ones(1,numel(out2.omegasX))/sqrt(3),'k--');
plot(1:numel(out2.omegasX),widthEY*ones(1,numel(out2.omegasX))/sqrt(3),'k--');
legend('omega X iterated','omega Y iterated','brute force values');
xlabel('iteration');ylabel('omegas');
hold off


figure(538);tiledlayout(2,3,'tilespacing','compact');colormap(gray);
% plot([width, width],[min(SUREall),max(SUREall)],':'); hold off;
t7 = nexttile;imagesc(b,[mm1 mm2]);title('blurry');
t3 = nexttile;imagesc(I,[mm1,mm2]);title('original');
t4 = nexttile;imagesc(rec1,[mm1,mm2]);title('estimated PSF Wiener');
t5 = nexttile;imagesc(rec2,[mm1,mm2]);title('estimated PSF BM3D')
t6 = nexttile;imagesc(rec4,[mm1,mm2]);title('best width PSF Wiener');

t8 = nexttile;imagesc(rec5,[mm1 mm2]);title('Wiener + BM3D');
linkaxes([t3 t4 t5 t6 t7 t8]);

fprintf('Estimated PSF Wiener + BM3D: %g\n',myrel(rec5,I));
fprintf('Estimated PSF BM3D error: %g\n',myrel(rec2,I));
fprintf('Estimated PSF Wiener error: %g\n',myrel(rec1,I));
fprintf('Best width PSF Wiener error: %g\n',myrel(rec4,I));
fprintf('blurry error: %g\n',myrel(b,I));
fprintf('true widthX: %g\n estimated widthX: %g\n',widthX,widthEX)
fprintf('true widthY: %g\n estimated widthY: %g\n',widthY,widthEY)
fprintf('lambda estimate: %g\n',lambdaE)
