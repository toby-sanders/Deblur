clear;
SNR = 20;
omega = 2;
lambdas = linspace(4,-1,10); 
lambdas = 10.^(-lambdas);% linspace(1e-4,1,100);
order = 2;
levels = 1;
angle = 0;
% type1 = 'gaus';
type = 'gaus';
rng(2021);
channel = 2;
sigma = 2.5/255;
addBlur = false;
% xx = 3101:3800;
% yy = 3201:3900;
% % xx = 2001:2500;
% % yy = 3601:4100;
% xx = 7501:8600;
% yy = 1:900;
% yy = 5601:6050;
% xx = 1901:2400;
% yy = 5001:5300;
% xx = 10241:10390;
xx = 501:1100;
yy = 6101:6650;

% get image and add noise
path = 'C:\Users\toby.sanders\Dropbox\TobySharedMATLAB\Surdex';
I0 = im2double(imread([path,filesep,'surdex1.tif']));
I = I0(yy,xx,channel);
[d1,d2] = size(I);
b = I;


if addBlur
    [~,hbAdd] = makeMotionPSF2D([d1,d2],2,2,0,'gaus');
    b = real(ifft2(fft2(b).*hbAdd));
end


V = my_Fourier_filters(order,levels,d1,d2,1);
% V = 200./abs(fft2(b)).^2;
Fb = fft2(b);

sigma = determineNoise1D(b,10);
opts.sigma = sigma;

opts.V = V;
opts.lambdas = lambdas;
% if strcmp(type,'tophat')
%     omegasX = 1:5;
%     omegasY = 1:5;
% else
    omegasX = linspace(1,4,10);
    omegasY = linspace(1,4,10);
% end
nX = numel(omegasX);
nY = numel(omegasY);

out = cell(nY,nX);
SUREall = zeros(nY,nX);
Fb = fft2(b);
tic;
for i = 1:nY
    for j = 1:nX
        [h,hhat] = makeMotionPSF2D([d1,d2],omegasX(j),omegasY(i),angle,type);
        out{i,j} = SURE_deblur(Fb,hhat,opts);
        SUREall(i,j) = out{i,j}.SUREbest;
    end
end
toc;

[~,ind] = min(SUREall(:));
[indY,indX] = ind2sub([nX,nY],ind);
widthEX = omegasX(indX);
widthEY = omegasY(indY);
lambdaE = out{ind}.lambdaBest;

% compare PSF widths found with the fixed point iteration
omega0 = (widthEX + widthEY)/sqrt(3)/2;
opts.tol = 1e-7;
opts.lambda = 8.4e2*sigma^2;
out2 = SURE_FP2(Fb,omega0,opts);


% evaluate improved BM3D solutions with recovered parameters
omegaEX = out2.omegasX(end);
omegaEY = out2.omegasY(end);
[h,hhat] = makeMotionPSF2D([d1,d2],omegaEX*sqrt(3),omegaEY*sqrt(3),angle,type);
filt1 = conj(hhat)./(abs(hhat).^2 + out2.lambdas(end).*opts.V);
rec1 = real(ifft2(Fb.*filt1));
BMopts.profile = 'fast';
rec2 = GBM3D_deconv(b,hhat,sigma,BMopts);
rec3 = LTsharpen(rec2,1.5,2/255);

opts.profile = 'default';
sigmaPSD = sigma^2*abs(filt1).^2;
rec5 = GBM3D(rec1,sigmaPSD,opts);



%%
Wresponse = abs(hhat.*filt1);
mm1 = 0.3;
mm2 = 1;
figure(537);tiledlayout(2,2,'tilespacing','compact');colormap(jet);
t1 = nexttile;hold off;
imagesc(omegasX,omegasY,log(max(SUREall,1e-10)));% hold on;
colorbar;xlabel('omegaX');ylabel('omegaY');title('SURE values');
t2 = nexttile;
imagesc(fftshift(Wresponse));colorbar;title('ideal response function')
t22 = nexttile;
imagesc(fftshift(h));
axis([round(d2/2)-9 round(d2/2)+10 round(d1/2)-9 round(d1/2)+10 ])
t23 = nexttile;hold off;
plot(out2.omegasX);hold on;
plot(out2.omegasY);
plot(1:numel(out2.omegasX),widthEX*ones(1,numel(out2.omegasX))/sqrt(3),'k--');
plot(1:numel(out2.omegasX),widthEY*ones(1,numel(out2.omegasX))/sqrt(3),'k--');
legend('omega X iterated','omega Y iterated','brute force values');
xlabel('iteration');ylabel('omegas');
hold off


figure(538);tiledlayout(2,2,'tilespacing','compact');colormap(gray);
% plot([width, width],[min(SUREall),max(SUREall)],':'); hold off;
% t7 = nexttile;imagesc(b,[mm1 mm2]);title('blurry');
t3 = nexttile;imagesc(I,[mm1,mm2]);title('original');
t4 = nexttile;imagesc(rec1,[mm1,mm2]);title('estimated PSF Wiener');
t5 = nexttile;imagesc(rec2,[mm1,mm2]);title('estimated PSF BM3D')
t8 = nexttile;imagesc(rec5,[mm1 mm2]);title('Wiener + BM3D');
linkaxes([t3 t4 t5 t8]);

fprintf('Estimated PSF Wiener + BM3D: %g\n',myrel(rec5,I));
fprintf('Estimated PSF BM3D error: %g\n',myrel(rec2,I));
fprintf('Estimated PSF Wiener error: %g\n',myrel(rec1,I));
fprintf('blurry error: %g\n',myrel(b,I));
fprintf('estimated widthX: %g\n',widthEX)
fprintf('estimated widthY: %g\n',widthEY)
fprintf('lambda estimate: %g\n',lambdaE)
fprintf('sigma estimate: %g\n',sigma*255)
