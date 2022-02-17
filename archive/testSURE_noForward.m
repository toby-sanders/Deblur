clear;
SNR =20;
omega = 2;
lambdas = linspace(4,-1,50); lambdas = 10.^(-lambdas);% linspace(1e-4,1,100);
order = 2;
levels = 1;
width = 3;
type = 'tophat';
epsilon = 1e-3;
rng(2022);


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
[h,hhat] = makeMotionPSF2D([d1,d2],width,width,0,type);
b = real(ifft2(fft2(I).*hhat));
[b,sigma] = add_Wnoise(b,SNR);

nL = numel(lambdas);
V = my_Fourier_filters(order,levels,d1,d2,1);
Fb = fft2(b);
hhat2 = abs(hhat).^2;
pert = randn(d1,d2);
Fpert = fft2(pert);



tr = zeros(nL,1);
ee = tr;
eeT = tr;

for i = 1:nL
    lambda = lambdas(i);
    f1 = real(ifft2(Fb./(hhat2 + lambda*V)));
    f2 = real(ifft2((Fb + epsilon*Fpert)./(hhat2 + lambda*V)));
    f = real(ifft2(fft2(f1).*conj(hhat)));
    tr(i) = sum(pert(:).*col(f2(:) - f1(:)))/epsilon;
    ee(i) = norm(f(:))^2 - 2*sum(b(:).*f1(:)) + 2*sigma^2*tr(i) + norm(b(:))^2;
    eeT(i) = norm(f(:) - I(:))^2;
end

opts.lambdas = lambdas;
opts.V = V;
opts.sigma = sigma;
opts.trueU = I;
out = SURE_deblur(Fb,hhat,opts);
out.SURE = out.SURE + norm(I(:))^2;
%%
offset = 1e4;
figure(109);
subplot(2,2,1);loglog(lambdas,real(ee) + offset);
subplot(2,2,2);loglog(lambdas,eeT);
subplot(2,2,3);hold off;
loglog(lambdas,sqrt(real(out.SURE))/norm(I(:)));hold on;
loglog(lambdas,sqrt(real(out.SURET))/norm(I(:)))