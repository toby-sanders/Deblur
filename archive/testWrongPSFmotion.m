clear;
SNR = 400;
omega = 2;
lambdas = linspace(4,-1,50); lambdas = 10.^(-lambdas);% linspace(1e-4,1,100);
alphas = linspace(2,15,15);
betas = linspace(0,2,11);
order = 2;
levels = 1;
width = 6;
type1 = 'tophat';
type2 = 'tophat';
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

[g,ghat] = makeMotionPSF([d1,d2],width,type1);
[h,hhat] = makeMotionPSF([d1,d2],width,type2);
b = real(ifft2(fft2(I).*ghat));
[b,sigma] = add_Wnoise(b,SNR);
V = my_Fourier_filters(order,levels,d1,d2,1);
Fb = fft2(b);


eeW = zeros(numel(lambdas),1);
for i = 1:numel(lambdas)
    lambda = lambdas(i);
    filt1 = conj(hhat)./(abs(hhat).^2 + lambda.*V);
    rec1 = real(ifft2(Fb.*filt1));
    eeW(i) = myrel(rec1,I);
end

[~,lambda] = min(eeW);
lambda = lambdas(lambda);

filt1 = hhat./(abs(hhat).^2 + lambda.*V);
rec1 = real(ifft2(Fb.*filt1));



%%
xx = linspace(0,1,100);
yy = xx./(xx.^2 + lambda);
% v = zeros(size(xx));
% for i = 1:numel(xx)
%     [~,mm] = min(abs(xx(i) - hhat(:,1)));
%     v(i) = V(mm,1);
% end
% yy2 = xx./(xx.^2 + lambda*v);
Wresponse = real(filt1.*ghat);

figure(777);
subplot(2,3,1);imagesc(real(fftshift(filt1.*ghat)));colorbar;title('Wiener response')
subplot(2,3,2);hold off;
plot(xx,yy);hold off;
legend('wiener filter');
subplot(2,3,3);hold off;
plot(xx,yy.*xx);hold off;
legend({'wiener response'},'location','southeast');
subplot(2,3,4);semilogx(lambdas,eeW);title('errors');xlabel('lambdas');
subplot(2,3,5);hold off;
plot(real(fftshift(Wresponse(:,1))));hold on;
legend('wiener response');


mm1 = 0;
mm2 = 1;
figure(778);colormap(gray);tiledlayout(2,2,'tilespacing','none');
t1 = nexttile;imagesc(I,[mm1,mm2]);
t2 = nexttile;imagesc(b,[mm1,mm2]);
t3 = nexttile;imagesc(rec1,[mm1,mm2]);
linkaxes([t1 t2 t3])


fprintf('Wiener error: %g\n',myrel(rec1,I));
fprintf('blurry error: %g\n',myrel(b,I));
fprintf('lambda: %g\n',lambda)