clear;
SNR = 20;
omega = 1.5;
lambdas = linspace(4,-1,20); 
lambdas = 10.^(-lambdas);% linspace(1e-4,1,100);
order = 2;
levels = 1;
iter = 40;
type = 'gaus';



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

[g,ghat] = makeGausPSF([d1,d2],omega);
b = real(ifft2(fft2(I).*ghat));
[b,sigma] = add_Wnoise(b,SNR);



V = my_Fourier_filters(order,levels,d1,d2,1);
% V = 200./abs(fft2(b)).^2;
Fb = fft2(b);

opts.sigma = sigma;
opts.V = V;
opts.order = order;
opts.lambdas = lambdas;
opts.tol = 1e-7;
bhat = fft2(b);
[h,hhat] = makeGausPSF([d1,d2],omega);
out = SURE_deblur(bhat,hhat,opts);
[U,out2] = SURE_FP_lambda(h,b,opts,sigma);
[U2,out3] = SURE_FP(bhat,hhat,opts);
SUREall = out.SUREbest;

lambda = out.lambdaBest;
out.lambdaBest
1/out2.mus(end)
out3.lambdas(end)
figure(99);hold off;
semilogy(real(out3.lambdas));hold on;
semilogy(real(1./out2.mus));