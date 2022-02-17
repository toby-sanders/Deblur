function A = getSuperResDeblurOpers(p,q,K,hhat,ghat)

if nargin<5
    g = zeros(p,q);
    g(1:K,1:K) = 1/K^2;
    g = fraccircshift(g,[-K/2 + 1/2, -K/2 + 1/2]);
    ghat = fft2(g);
end
ghat = ghat.*hhat;

A = @(U,mode)SuperForward(U,mode,p,q,ghat,K);
% Dt = @(U)SuperTranspose(U,p,q,ghat,K);

function U = SuperForward(U,mode,p,q,ghat,K)
    switch mode
        case 1
            U = reshape(U,p,q);
            U = (ifft2(fft2(U).*ghat));
            U = U(1:K:end,1:K:end);
            U = U(:);
        case 2
            U = reshape(U,p/K,q/K);
            U2 = zeros(p,q);
            U2(1:K:end,1:K:end) = U;
            U = (ifft2(fft2(U2).*conj(ghat)));
            U = U(:);
    end
