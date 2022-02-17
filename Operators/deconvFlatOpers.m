function y = deconvFlatOpers(x,mode,hhat,S,gamma)

[p,q] = size(hhat);
switch mode
    case 1        
        x = reshape(x,p,q);
        y = ifft2(fft2(x).*hhat);
        y = [y(:);gamma*x(S)];
    case 2
        b1 = reshape(x(1:p*q),p,q);
        y = zeros(p,q);
        y(S) = gamma*x(p*q+1:end);
        y = y + ifft2(fft2(b1).*conj(hhat));
        y = y(:);
end