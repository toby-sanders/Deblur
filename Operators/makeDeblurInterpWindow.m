function w = makeDeblurInterpWindow(a,b,k)

% make interpolation windows for stitching image patches back together

% Written by Toby Sanders @Lickenbrock Tech
% 3-17-2021

% inputs (a,b) are the image patch size, (k) is the overlap size
w = ones(a,b);
% ramp = linspace(.01,1,k);
ramp = linspace(-10,10,k);
ramp = 1./(1+exp(-ramp));


w(1:k,:) = w(1:k,:).*repmat(ramp',1,b);
w(:,1:k) = w(:,1:k).*repmat(ramp,a,1);
w(end-k+1:end,:) = w(end-k+1:end,:).*flipud(repmat(ramp',1,b));
w(:,end-k+1:end) = w(:,end-k+1:end).*fliplr(repmat(ramp,a,1));