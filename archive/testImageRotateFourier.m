clear;
d = 128;
rangle = 90;

P = zeros(d);
P(d/2-9:d/2+10,d/2-29:d/2+30) = 1;
P = phantom(d);
% P2 = imrotate(P,rangle,'bilinear','crop');
P2 = P';

FP = fft2(P);
FP2 = fft2(P2);

FP3 = circshift(transpose(FP2),[d/2,d/2]);
% FP3 =  imrotate(circshift(FP2,[d/2-1,d/2]),-rangle,'bilinear','crop');
figure(54);tiledlayout(2,2);
t1 = nexttile;imagesc(log(abs(fftshift(FP))));
t2 = nexttile;imagesc(log(abs(FP3)));
t3 = nexttile;imagesc(log(abs(imag(fftshift(FP)))));
t4 = nexttile;imagesc(log(abs(imag(FP3))));
linkaxes([t1 t2 t3 t4])

D = angle(ifftshift(FP)./FP3);
myrel(ifftshift(FP),FP3)
myrel(abs(ifftshift(FP)),abs(FP3))