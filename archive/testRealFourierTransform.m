clear;
d = 16;

% this process should generate a random signal that has a real valued
% fourier transform
x = rand(d,1);
x = 1:d;
if mod(d,2)==0
    for i =d/2+2:d
        % x(i + d/2 +1) = x(i+1);
        x(i) = x(d-i+2);
    end
else
    for i = (d+1)/2:d
        x(i) = x(d-i+2);
    end
end
y = fft(x)


figure(75);
subplot(1,2,1);hold off;
plot(fftshift(abs(real(y))));hold on;
plot(fftshift(abs(imag(y))));
axis([0 d -1 max(abs(y))])
subplot(1,2,2);
plot(x);