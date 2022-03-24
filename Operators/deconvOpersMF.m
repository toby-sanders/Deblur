function y = deconvOpersMF(x,mode,hhat)

[p,q,nF] = size(hhat);
switch mode
    case 1        
        x = reshape(x,p,q);
        y = ifft2(fft2(x).*hhat);
        y = y(:);
    case 2
        x = reshape(x,p,q,nF);
        y = sum(ifft2(fft2(x).*conj(hhat)),3);
        y = y(:);
end