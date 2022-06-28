function y = deconvOpers1DZ(x,mode,hhat)

[p,q] = size(hhat);
switch mode
    case 1        
        x = reshape(x,p,q);
        y = ifft(fft(x,[],2).*hhat,[],2);
        y = y(:);
    case 2
        x = reshape(x,p,q);
        y = ifft(fft(x,[],2).*conj(hhat),[],2);
        y = y(:);
end