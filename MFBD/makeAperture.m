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
%
% makeAperture:
%  Matlab function to create a binary aperture function.
%
% Usage:
%  A = makeAperture(physicalParams);
%
% Inputs:
%  physicalParams: (physical parameters for the system)
%
% Output:
%  A = N1xN2 binary apterture function
%
% Coded by Timothy J. Schulz
%  for Lickenbrock Technologies, INC
%  July 6, 2017
%
% Updates:
%  11/29/2018 (MAA)
%     Modified the code to represent the aperture in the frequency domain following below equations.
%     D/W = D*dx/lambda
%     dx = dxs/f
%     D/W = Ddxs / f*lambda
%     dus = 1/N*dxs
%     Us = N*dus = 1/dxs
%     BWs = 2*D / f*lambda = 4*NA / lambda
%     BWs/Us = 4*NA / lambda *dxs
%
% input variables
% physicalParams.lambda = (out{5})/(10^9); % nominal wavelength for the light (meters) converted to nm 
% physicalParams.D      = out{1}; % main aperture diameter (meters)
% physicalParams.Do     = physicalParams.D*0.2; % aperture obscurration diameter (meters)
% physicalParams.N      = N; % detector array size (2 elements)
% physicalParams.dx     = out{4}; % detector sample spacing (nano radians per sample)
%  
function aperture = makeAperture_transition(physicalParams,object)

  if physicalParams.Do > physicalParams.D
    error('obscurration cannot be bigger than aperture');
  end
  %
  % determine the size of the aperture array
  %
  if physicalParams.Do > physicalParams.D
    error('obscurration cannot be bigger than aperture');
  end
  %
  % determine the size of the aperture array
  %
  n_up     = physicalParams.samplingFactor;
  W        = physicalParams.lambda/physicalParams.dx;
  N1       = physicalParams.N(1);
  N2       = physicalParams.N(2);
  D        = physicalParams.D;
  Do       = physicalParams.Do;
  X        = linspace(-W/2, W/2, N2);
  Y        = linspace(-W/2, W/2, N1);  
  [x1, x2] = meshgrid(X*n_up,Y*n_up);
  r        = sqrt(x1.^2 + x2.^2);
  
  % space domain
%   aperture.A = (r > Do/2) & (r < D/2);
%   aperture.A = double(fftshift(aperture.A));
%   aperture.A = N*aperture.A/sqrt(sum(aperture.A(:).^2));
   aperture.x = X;
   aperture.r = r;
 
  % frequency domain  --- Cycles/meter
  f         = 1.0; % focal length set to 1 for normalized spatial samples
  dxs       = physicalParams.dx.*((10^9)).*f; 
  %   dus       = 1./N.*dxs ;
  Us        = 1./dxs;
  Bws_o     = (D)/(f.*physicalParams.lambda.*(10^9));
  Bws_i     = (Do)/(f.*physicalParams.lambda.*(10^9));

  u_X       = linspace(-(Us/2), Us/2, N2);
  u_Y       = linspace(-(Us/2), Us/2, N1);
  [u1, u2]  = meshgrid(u_X*n_up,u_Y*n_up);
  r_u       = sqrt(u1.^2 + u2.^2);
  

  aperture.A = (r_u > Bws_i/2)& (r_u < Bws_o/2);
  aperture.A = fftshift(aperture.A);
  % aperture.A = physicalParams.N(1)*aperture.A/sqrt(sum(aperture.A(:).^2));

 
end