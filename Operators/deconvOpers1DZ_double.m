function y = deconvOpers1DZ(x,mode,hhat)

[p,q] = size(hhat);
switch mode
    case 1        
        x = reshape(x,p,q);
        xhat = fft(x,[],2);
        y1 = ifft(xhat.*hhat,[],2);
        y2 = ifft(xhat.*conj(hhat),[],2);
        y = [y1(:);y2(:)];
    case 2
        x = reshape(x,p,q,2);
        xhat = fft(x,[],2);
        y1 = ifft(xhat(:,:,1).*conj(hhat),[],2);
        y2 = ifft(xhat(:,:,2).*hhat,[],2);
        y = y1(:) + y2(:);
end