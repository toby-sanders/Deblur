clear;
d = 16;

% this process should generate a random signal that has a real valued
% fourier transform
x = zeros(d,d);
for i = 2:d
    for j = 2:d
        x(i,j) = i+j + randn(1);
    end
end
x = makeGausPSF(d,3,1,15,1);

x2 = x;
if mod(d,2)==0
    x(d/2+1:d,d/2+1:d) = fliplr(flipud(x(2:d/2+1,2:d/2+1)));
    x(d/2+2:d,2:d/2) = flipud(fliplr(x(2:d/2,d/2+2:d)))
    % x(d/2+2:end,1) = flipud(x(2:d/2,1));
    % x(1,d/2+2:end) = fliplr(x(1,2:d/2));
elseif 0==1
    for i = (d+1)/2:d
        x(i) = x(d-i+2);
    end
end
% x = flip_F_Fourier(x);
y = fft2(x2);
myrel(x,x2)

figure(77);subplot(2,2,1);
imagesc(fftshift(abs(real(y))));colorbar;
subplot(2,2,2);
imagesc(fftshift(abs(imag(y))));colorbar;
subplot(2,2,3);
imagesc(x2);
subplot(2,2,4);
imagesc(x)
title(sprintf('rel diff: %g',myrel(x,x2)));
