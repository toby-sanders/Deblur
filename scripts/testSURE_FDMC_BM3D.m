% testing the SURE approximation whenever the a finite difference monte
% carlo is used to estimate the trace term. Testing with BM3D 

clear;
SNR = 10; % SNR in noisy image
N = 15; % number of values to test
lambdas = linspace(.5,2,N); % scaling test values
epsilon = 0.001; % perturbation scale for finite difference approximation

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
[b,sigma] = add_Wnoise(I,SNR);



% initialize variables
ee = zeros(N,2);
rec = zeros(d1,d2,N);
recD = zeros(d1,d2,N);
SURE = zeros(N,1);MSE = SURE;
tr = zeros(N,1);
er = zeros(N,1);
pert = randn(d1,d2);

% loop over scaling factors
opts.profile = 'default';
for i = 1:N
    rec(:,:,i) = GBM3D(b,sigma*lambdas(i),opts); % denoise
    recD(:,:,i) = GBM3D(b + epsilon*pert,sigma*lambdas(i),opts); % denoise with perterbation
    tr(i) = sum(pert(:).*col(recD(:,:,i) - rec(:,:,i)))/epsilon; % trace approximation with FD
    er(i) = norm(col(rec(:,:,i) - b))^2;

    % SURE estimator and true MSE
    SURE(i) = sum(col(rec(:,:,i)-b).^2) + 2*sigma^2*tr(i) - d1*d2*sigma^2;
    MSE(i) = sum(col(rec(:,:,i) - I).^2);
end


%% plot results
figure(177);
hold off;
plot(lambdas,SURE);hold on;
plot(lambdas,MSE);
hold off;
xlabel('scaling factor');
ylabel('MSE/SURE');
legend('SURE','MSE')



