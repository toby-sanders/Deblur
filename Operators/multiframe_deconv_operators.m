function x = multiframe_deconv_operators(hhat,nF,x,mode,p,q)

% Written by Toby Sanders @ Lickenbrock Tech.
% 10/14/2019

% this code computes forward (mode 1) and adjoint (mode 2) operators needed
% for subpixel resolution imaging from multiple frames

% inputs
% p,q - dimensions of the image to be reconstructed (3 times the data)
% hhat - FT of bluring kernel, h, which should be 3x3 block
% nF - number of frames, so data is size p/3 x q/3 x nF


switch mode
    case 1
        % blur, downsample, then replicate nF times
        x = reshape(x,p,q);
        x = ifft2(fft2(x).*hhat);
        x = x(2:3:end,2:3:end);
        x = x(:);
        x = repmat(x,nF,1);
    case 2
        % upsample, conj. blur, then sum over frames
        x = reshape(x,p/3,q/3,nF);
        y = zeros(p,q,nF);
        y(2:3:end,2:3:end,:) = x;
        x = sum(ifft2(fft2(y).*conj(hhat)),3);
        x = x(:);
end