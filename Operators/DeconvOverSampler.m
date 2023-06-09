function x = DeconvSampler(x,mode,hhat,S,p,q)

switch mode
    case 1
        x = reshape(x,p,q);
        x = (ifft2(fft2(x).*hhat));
        x = S*x(:);
    case 2
        x = reshape(S'*x(:),p,q);
        x = ifft2(fft2(x).*conj(hhat));
        x = (x(:));
end

        
        