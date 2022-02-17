function x = DeconvSampler(x,mode,hhat,S,p,q,nF)

switch mode
    case 1
        x = reshape(x,p,q);
        x = (ifft2(fft2(x).*hhat));
        x = x(S);
    case 2
        y = zeros(p,q);
        y(S) = x;
        x = ifft2(fft2(y).*conj(hhat));
        x = (x(:));
end

        
        