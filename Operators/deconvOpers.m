function y = deconvOpers(x,mode,hhat)

[p,q] = size(hhat);
switch mode
    case 1        
        x = reshape(x,p,q);
        y = ifft2(fft2(x).*hhat);
        y = y(:);
    case 2
        x = reshape(x,p,q);
        y = ifft2(fft2(x).*conj(hhat));
        y = y(:);
end