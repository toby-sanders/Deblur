function [h,hhat] = makeMotionBlur(n,L,theta,nZ,postBlur)

% make a "one-dimensional" motion blur point spread function,
% it is assumed that the motion is constant over the exposure time,
% resulting in a binary PSF.
% INPUTS: 
% n is the image dimension
% L is the length of the bluring kernel
% theta is the angle of the motion

if numel(n)==1, d1 = n; d2 = n;
else d1 = n(1); d2 = n(2); 
end
if nargin<5, postBlur = false; end

h = zeros(d1,d2);
h(1,1:L) = 1;
h = circshift(h,[0 -floor(L/2)]);
h = imrotate(fftshift(h),theta,'crop');
[~,ghat] = makeGausPSF([d1,d2],1);
h0 = real(ifft2(fft2(h).*ghat));
h0 = h0 + [rand(d1,d2)-.5]*max(h0(:))/2;
[~,S] = sort(h0(:));
h = zeros(d1,d2);
h(S(end-nZ+1:end)) = h0(S(end-nZ+1:end));
if postBlur
    [~,ghat] = makeGausPSF([d1,d2],.5);
    h = real(ifft2(fft2(h).*ghat));
end


h = ifftshift(h/sum(h(:)));
hhat = fft2(h);