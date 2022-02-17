function [R,d0,B] = getDeblur_distribution(m,n,d0)

% set up the parameters for decomposing an image into pieces to process a
% spatially varying PSF estimation/deconvolution

% Written by Toby Sanders @Lickenbrock Tech
% 12-21-21

if nargin<3
    % d0 is maximum size of each image patch
    d0 = 256;
end

B = 16; % B is the size of the overlap between patches
cntY = round(m/(d0-B)); % number of patches we'll have in y-axis
cntX = round(n/(d0-B)); % number of patches we'll have in x-axis
R = zeros(4,cntX,cntY); % R stores the indices for all patches

% loop over all image patches and save indices
% each index moves d0-B pixels forward
for i = 1:cntY
    for j = 1:cntX
        indX1 = (j-1)*d0+1-B*(j-1);
        indX2 = indX1 + d0 - 1;
        indY1 = (i-1)*d0+1-B*(i-1);
        indY2 = indY1 + d0 - 1;
        R(:,j,i) = [indY1;indY2;indX1;indX2];
    end
end

% set the last patches to the maximum of the original image size, since we
% most likely spilled over the edges (or didn't reach the edge)
R(2,:,end) = m;
R(4,end,:) = n;
