function [h,hhat] = randBlur(n,L,theta)


if numel(n)==1, d1 = n; d2 = n;
else d1 = n(1); d2 = n(2); 
end

x = round(L*cosd(theta));
y = round(L*sind(theta));
h = zeros(L);
h(1:L+1:end) = 1;
h = imresize(h,[y,x],'bilinear');
h = h + rand(y,x);
[~,hhat] = makeGausPSF([y,x],.4);
h = real(ifft2(fft2(h).*hhat));


hhat = fft2(h);