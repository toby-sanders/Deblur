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
% written by Tim Schulz
% used for deconvolution 

 function phaseEstimate = phaseRetrieval(data, aperture, initialPhaseEstimate)

  phaseEstimate = initialPhaseEstimate;
  
  dataSQRT = sqrt(data);
  for iter=1:100
    g = ifft2(aperture.A.*exp(1j*phaseEstimate));
    G = fft2(dataSQRT.*exp(1j*angle(g)));
    phaseEstimate = angle(G);
  end 
end