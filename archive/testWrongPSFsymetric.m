clear;
SNR =20;
omega = 2;
lambdas = linspace(4,-1,50); lambdas = 10.^(-lambdas);% linspace(1e-4,1,100);
order = 2;
levels = 1;
width = 3;
type1 = 'tophat';
type2 = 'gaus';
rng(2021);


% get image and add noise
path = 'C:\Users\toby.sanders\Dropbox\archives\data\testImages\';
% path = '/home/tobysanders/Dropbox/archives/data/testImages/';
% I = im2double(rgb2gray(imread([path,'lena.png'])));
% I = im2double((imread([path,'house.tif'])));I = I(:,:,1);
% I = im2double(rgb2gray(imread([path,'peppers2.png'])));
% I = im2double((imread([path,'confFlag.tif'])));
% I = im2double((imread([path,'SeasatDeblurred.png'])));
% I = im2double(rgb2gray(imread([path,'PineyMO_Oimage.tif'])));
I = im2double(rgb2gray(imread([path,'monarch.png'])));
% I = im2double((imread('cameraman.tif')));
% I = phantom(512);

[d1,d2] = size(I);

[g,ghat] = makeSymPSF([d1,d2],width,type1);
[h,hhat] = makeSymPSF([d1,d2],width,type2);
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

opts.profile = 'fast';
rec2 = GBM3D_deconv(b,hhat,sigma,opts);
rec3 = LTsharpen(rec2,1.5,2/255);
%%
xx = linspace(0,1,100);
yy = xx./(xx.^2 + lambda);
% v = zeros(size(xx));
% for i = 1:numel(xx)
%     [~,mm] = min(abs(xx(i) - hhat(:,1)));
%     v(i) = V(mm,1);
% end
% yy2 = xx./(xx.^2 + lambda*v);
epsilon = 1e2;
Wresponse = real(filt1.*ghat);
Wresponse = real(fft2(rec1)./fft2(I));
BMresponse = real(fft2(rec2)./fft2(I));
Sresponse = real(fft2(rec3)./fft2(I));
psfE1 = abs(fft2(b)./(fft2(rec1) + epsilon));
psfE2 = abs(fft2(b)./(fft2(rec2) + epsilon));
psfE3 = abs(fft2(b)./(fft2(I)+epsilon));
% psfE = real(ifft2(fft2(b)./fft2(I)));

figure(777);
subplot(3,3,1);imagesc(fftshift(Wresponse),[0 2]);colorbar;title('Wiener response')
subplot(3,3,2);hold off;
% plot(xx,yy);hold off;
% legend('wiener filter');
imagesc(fftshift(Sresponse),[0 2]);
colorbar;title('deconv + sharpening response')
subplot(3,3,3);hold off;
plot(xx,yy.*xx);hold off;
legend({'wiener response'},'location','southeast');
subplot(3,3,4);% semilogx(lambdas,eeW);title('errors');xlabel('lambdas');
% imagesc(fftshift(psfE));colorbar;
imagesc(fftshift(abs(hhat)));
subplot(3,3,5);hold off;
plot(real(fftshift(Wresponse(:,1))));hold on;
legend('wiener response');
subplot(3,3,6);
imagesc(fftshift(BMresponse),[0 2]);
colorbar;title('BM3D response')
subplot(3,3,7);imagesc(fftshift(psfE1),[0 1]);
subplot(3,3,8);imagesc(fftshift(psfE2),[0 1]);
subplot(3,3,9);imagesc(fftshift(psfE3),[0 1]);

mm1 = 0;
mm2 = 1;
figure(778);colormap(gray);tiledlayout(2,3,'tilespacing','none');
t1 = nexttile;imagesc(I,[mm1,mm2]);title('original');
t2 = nexttile;imagesc(b,[mm1,mm2]);title('blurry');
t3 = nexttile;imagesc(rec1,[mm1,mm2]);title('Wiener');
t4 = nexttile;imagesc(rec2,[mm1,mm2]);title('BM3D deconv');
t5 = nexttile;imagesc(rec3,[mm1,mm2]);title('BM3D + sharpening');
linkaxes([t1 t2 t3 t4 t5])

fprintf('BM3D error: %g\n',myrel(rec2,I));
fprintf('sharpened error: %g\n',myrel(rec3,I));
fprintf('Wiener error: %g\n',myrel(rec1,I));
fprintf('blurry error: %g\n',myrel(b,I));
fprintf('lambda: %g\n',lambda)