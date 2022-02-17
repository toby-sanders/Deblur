function [h,hhat] = makeMotionPSF2D(n,widthX,widthY,angle,type)


if numel(n)==1, d1 = n; d2 = n;
elseif numel(n)==2, d1 = n(1); d2 = n(2);
else, error('too many dimensions in n');
end

if strcmpi(type,'tophat')
    widthYR = floor(widthY);
    widthXR = floor(widthX);
    h = zeros(d1,d2);
    h([1:widthYR,end-widthYR+2:end],[1:widthXR,end-widthXR+2:end]) = 1;    
    if widthX~=widthXR
        h([1:widthYR,end-widthYR+2:end],[widthXR+1,end-widthXR+1]) = widthX-widthXR;
    end
    if widthY~=widthYR
        h([widthYR+1,end-widthYR+1],[1:widthXR,end-widthXR+2:end]) = widthY-widthYR;
    end
elseif strcmpi(type,'gaus')
    [h,~] = makeGausPSF2D([d1,d2],widthX/sqrt(3),widthY/sqrt(3));
elseif strcmpi(type,'laplace')
    [h,~] = makeLaplacePSF2D([d1,d2],widthX*sqrt(2/3),widthY*sqrt(2/3));
end
h = ifftshift(imrotate(fftshift(h),angle,'bilinear','crop'));
h = h/sum(h(:));
hhat = fft2(h);