clear;
SNR = 15;
sigmaFactors = [.75,1,1.5];
N = numel(sigmaFactors);
epsilon = 1e-3;
rng(2021);
realData = false;

% get image and add noise
path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\';
% path = '/home/tobysanders/Dropbox/archives/data/testImages/';
I = im2double(rgb2gray(imread([path,'lena.png'])));
% I = im2double(imread([path,'soccerField.tif']));
% I = im2double((imread([path,'house.tif'])));I = I(:,:,1);
% I = im2double(rgb2gray(imread([path,'peppers2.png'])));
% I = im2double((imread([path,'confFlag.tif'])));
% I = im2double((imread([path,'SeasatDeblurred.png'])));
% I = im2double(rgb2gray(imread([path,'PineyMO_Oimage.tif'])));
% I = im2double(rgb2gray(imread([path,'monarch.png'])));
% I = im2double((imread('cameraman.tif')));
% I = phantom(512);

[d1,d2] = size(I);

if ~realData
    [b,sigma] = add_Wnoise(I,SNR);
else
    b = I;
    sigma = 7/255;
end

pert = randn(d1,d2);

rec = zeros(d1,d2,N);
recP = zeros(d1,d2,N);
recH = zeros(d1,d2,N);
c = zeros(N,1);
d = zeros(N,1);



% T = [0 -1 0; -1 4 -1; 0 -1 0];
[h,hhat] = makeGausPSF([d1,d2],2,1);

opts.profile = 'default';
for i = 1:N
    rec(:,:,i) = GBM3D(b,sigma*sigmaFactors(i),opts);
    % recH(:,:,i) = imfilter(rec(:,:,i),T,'replicate');
    recH(:,:,i) = real(ifft2(fft2(rec(:,:,i)).*hhat));
    recP(:,:,i) = GBM3D(b + pert*epsilon,sigma*sigmaFactors(i),opts);
    c(i) = sum(col(rec(:,:,i).*b));
    d(i) = -sigma^2*sum(col(pert.*(recP(:,:,i) - rec(:,:,i))))/epsilon;
end


%%
[~,which] = min(abs(1-sigmaFactors));
rec(:,:,N+1) = LTsharpen(rec(:,:,which),1,1/255);
recP(:,:,N+1) = LTsharpen(recP(:,:,which),1,1/255);
recH(:,:,N+1) = real(ifft2(fft2(rec(:,:,N+1)).*hhat));
c(N+1) = sum(col(rec(:,:,N+1).*b));
d(N+1) = -sigma^2*sum(col(pert.*(recP(:,:,N+1) - rec(:,:,N+1))))/epsilon;
N = N+1;
%%
A = zeros(N,N);
B = zeros(N,N);
gamma = 1e-1;
for i = 1:N
    for j = 1:N
        A(i,j) = sum(col(rec(:,:,i).*rec(:,:,j)));
        B(i,j) = gamma*sum(col(recH(:,:,i).*recH(:,:,j)));
    end
end

% A = A+B;



alpha = A\(c+d);
alpha2 = cgs(A,c+d,1e-5);
alpha3 = basic_GDNN(A,c+d,N,1,1,20);
alpha4 = basic_GD(A,c+d,N,1,1,20);
recF = zeros(d1,d2);
recF2 = recF;
recF3 = recF;
recF4 = recF;
for i = 1:N
    recF = recF + alpha(i)*rec(:,:,i);
    recF2 = recF2 + alpha2(i)*rec(:,:,i);
    recF3 = recF3 + alpha3(i)*rec(:,:,i);
    recF4 = recF4 + alpha4(i)*rec(:,:,i);
end


mm1 = 0.0;
mm2 = 1;
recB = rec(:,:,which);
figure(171);tiledlayout(2,3,'tilespacing','compact');colormap(gray);
t1 = nexttile;imagesc(recB,[mm1 mm2]);
title(sprintf('Basic BM3D, PSNR = %g',myPSNR(I,recB)));
t2 = nexttile;imagesc(recF2,[mm1 mm2]);
title(sprintf('CGS coef, PSNR = %g',myPSNR(I,recF2)));
t3 = nexttile;imagesc(recF3,[mm1 mm2]);
title(sprintf('GD NN coef, PSNR = %g',myPSNR(I,recF3)));
t4 = nexttile;imagesc(recF,[mm1 mm2]);
title(sprintf('QR coef, PSNR = %g',myPSNR(I,recF)));
t5 = nexttile;imagesc(I,[mm1 mm2]);title('original');
t6 = nexttile;imagesc(b,[mm1 mm2]);title('noisy');
linkaxes([t1 t2 t3 t4 t5 t6]);

figure(172);tiledlayout(2,2);
t5 = nexttile;hold off;
plot(alpha);
t6 = nexttile;
plot(alpha2);
t7 = nexttile;
plot(alpha3);
t8 = nexttile;
plot(alpha4);



