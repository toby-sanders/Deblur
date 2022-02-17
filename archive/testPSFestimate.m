clear;
SNR = 50;
omega = 2;
lambdas = linspace(4,-1,50); lambdas = 10.^(-lambdas);% linspace(1e-4,1,100);
order = 2;
levels = 1;
width = 2;
type1 = 'gaus';
type2 = 'laplace';
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

[d1,d2] = size(I);

[g,ghat] = makeSymPSF([d1,d2],width,type1);
[h,hhat] = makeSymPSF([d1,d2],width,type2);
b = real(ifft2(fft2(I).*ghat));
[b,sigma] = add_Wnoise(b,SNR);






V = my_Fourier_filters(order,levels,d1,d2,1);
% V = 200./abs(fft2(b)).^2;
Fb = fft2(b);

opts.sigma = sigma;
opts.V = V;
opts.lambdas = lambdas;
if strcmp(type2,'tophat')
    omegas = 1:10;
else
    omegas = linspace(1,width*1.5,15);
end

out = cell(numel(omegas),1);
SUREall = zeros(numel(omegas),1);
for i = 1:numel(omegas)
    [h,hhat] = makeSymPSF([d1,d2],omegas(i),type2);
    out{i} = SURE_deblurF(b,hhat,opts);
    SUREall(i) = out{i}.SUREbest;
end

[~,ind] = min(SUREall);
widthE = omegas(ind);
lambdaE = out{ind}.lambdaBest;
[h,hhat] = makeSymPSF([d1,d2],widthE,type2);
filt1 = hhat./(abs(hhat).^2 + lambdaE.*opts.V);
rec1 = real(ifft2(Fb.*filt1));
BMopts.profile = 'fast';
rec2 = GBM3D_deconv(b,hhat,sigma,BMopts);
rec3 = LTsharpen(rec2,1.5,2/255);


[f,fhat] = makeSymPSF([d1,d2],width,type2);
filt2 = fhat./(abs(fhat).^2 + lambdaE.*opts.V);
rec4 = real(ifft2(Fb.*filt2));

opts.profile = 'default';
sigmaPSD = sigma^2*abs(filt1).^2;
rec5 = GBM3D(rec1,sigmaPSD,opts);



%%
mm1 = 0;
mm2 = 1;
Wresponse = real(filt1.*ghat);
figure(537);tiledlayout(3,3,'tilespacing','compact');colormap(gray);
t1 = nexttile;hold off;
plot(omegas,SUREall);hold on;
plot([width, width],[min(SUREall),max(SUREall)],':'); hold off;
t7 = nexttile;imagesc(b,[mm1 mm2]);title('blurry');
t3 = nexttile;imagesc(I,[mm1,mm2]);title('original');
t4 = nexttile;imagesc(rec1,[mm1,mm2]);title('estimated PSF Wiener');
t5 = nexttile;imagesc(rec2,[mm1,mm2]);title('estimated PSF BM3D')
t6 = nexttile;imagesc(rec4,[mm1,mm2]);title('best width PSF Wiener');
t2 = nexttile;imagesc(fftshift(Wresponse));
title('response function');colorbar;
t8 = nexttile;imagesc(rec5,[mm1 mm2]);title('Wiener + BM3D');
linkaxes([t3 t4 t5 t6 t7 t8]);

fprintf('Estimated PSF Wiener + BM3D: %g\n',myrel(rec5,I));
fprintf('Estimated PSF BM3D error: %g\n',myrel(rec2,I));
fprintf('Estimated PSF Wiener error: %g\n',myrel(rec1,I));
fprintf('Best width PSF Wiener error: %g\n',myrel(rec4,I));
fprintf('blurry error: %g\n',myrel(b,I));
fprintf('true width: %g\n estimated width: %g\n',width,widthE)
fprintf('lambda estimate: %g\n',lambdaE)
