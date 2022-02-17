%
%  ============================================================================
%
%  Copyright (C) Lickenbrock Technologies LLC  2017 -
%
%  The contents of this file are the sole and propriety property of Lickenbrock
%      Technologies, LLC (Lickenbrock), and can only be used with the express
%      permission of Lickenbrock.
%
%  If you did not receive this file with the express permission of Lickenbrock,
%      you are obligated to remove this file completely from your system(s) and
%      contact Lickenbrock directly for permission to obtain and use this
%      file.
%
%  Visit http://www.lickenbrock.com for contact information.
%
%  ============================================================================
%
%  Limitation of Liability Notice:
%      This file is provided on an "AS IS" basis, without any other warranties
%      or conditions, express or implied, including, but not limited to,
%      warranties of merchantability, satisfactory quality, fitness for a
%      particular purpose, or those arising by law, statute, usage of trade,
%      course of dealing, or otherwise.
%
%  ============================================================================
% 
%
% makePSF:
%  Matlab function to create a point-spread function from an aperture and
%  phase screen.
%
% Usage:
%  outputs = makePSF(A, phaseScreen, physicalParams);
% Inputs:
%  A           = aperture function (positive values <= 1)
%  phaseScreen = phase screen (radians)
%  physicalParams: (physical parameters for the system)
%
% Output:
%  (results of point-spread function generation)
%    outputs.psf = point-spread function
%    outputs.H   = frequency response
%    outputs.y   = spatial index (meters)
%
% Coded by Timothy J. Schulz
%  for Lickenbrock Technologies, INC
%  June 26, 2017
%

function outputs = makePSF(A, phaseScreen, physicalParams)

%   lambda   = physicalParams.lambda;
%   W        = physicalParams.lambda/physicalParams.dx;
%   N        = physicalParams.N(1);
%   dx       = W/N;
%   dy       = lambda/(N*dx);
%   y        = [-N/2:1:N/2-1]*dy;
 
  %   psf =abs((ifft2(A.*exp(1j*phaseScreen)))).^2;  %fftshift added 
  ctf               = A.*exp(1j*phaseScreen);
  psf_coh           = ifft2(ctf);
  psf               = abs(psf_coh.^2);
  %normalize to sum to 1
  psf = psf ./sum(sum(psf));
  %save out psf
  outputs.psf = psf;
  outputs.H   = fft2(psf);  
 
end