function [h,hhat] = makeMotionPSF(n,width,type)


if numel(n)==1, d1 = n; d2 = n;
elseif numel(n)==2, d1 = n(1); d2 = n(2);
else, error('too many dimensions in n');
end

if strcmpi(type,'tophat')
    widthR = floor(width);
    h = zeros(d1,d2);
    h(1:widthR,1) = 1;
    h(end-widthR+2:end,1) = 1;
    if widthR~=width
        h([widthR+1,end-widthR+1],1) = width-widthR;
    end
    h = h/sum(h(:));
    hhat = fft2(h);
elseif strcmpi(type,'triangle')
    width = round(width*1.25);
    h = zeros(d1,d2);
    h(1:width+1,1) = linspace(1,0,width+1);
    h(end-width+1:end,1) = linspace(0,h(2),width);
    h = h/sum(h(:));
    hhat = fft2(h);
elseif strcmpi(type,'gaus')
    [h,hhat] = makeGausPSF2D([d1,d2],.1,width/sqrt(3));
elseif strcmpi(type,'laplace')
    [h,hhat] = makeLaplacePSF2D([d1,d2],.1,width/sqrt(3));
end
