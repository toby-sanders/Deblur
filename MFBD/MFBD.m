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

function [processingParams,out_psf, DeblurCancelled, diagnostic]= multiFrameBlindDeblur(data,phase,bg,aperture,...
    physicalParams, processingParams,gpuID) % ,Prog_bar
% set up GPU variables for MFBD
 % same variables as those on CPU
 if nargin<7
    gpuID       = gpuDevice(2);
    reset(gpuID);
 end
 
% Added as part of DB-7
% Checks to see if the size of the input data is greater than the amount of
% available GPU memory after it is freed up by the reset(gpuID) command -

% Check what type the input data is, this determines the byte size of the
% data
if ((class(data) == "uint8") || (class(data) == "int8"))
    DataByteSize = 1;
elseif ((class(data) == "uint16") || (class(data) == "int16"))
    DataByteSize = 2;
elseif ((class(data) == "uint32") || (class(data) == "int32"))
    DataByteSize = 8;
elseif ((class(data) == "uint64") || (class(data) == "uint64"))
    DataByteSize = 16;
else
    DataByteSize = 16;
    % errordlg('This data type is not currently supported.');
    % return;
end
%check avaible gpu memory 
if processingParams.superRes
    RequiredMemory_blockDependent = size(data,1)*3*size(data,2)*3*size(data,3)*DataByteSize*4;
    RequiredMemory_frameDependent= size(data,1)*3*size(data,2)*3*DataByteSize*4;%if data size not power of 2 tends to have 8 times more space needed
else
    RequiredMemory_blockDependent = size(data,1)*size(data,2)*size(data,3)*DataByteSize;
    RequiredMemory_frameDependent= size(data,1)*size(data,2)*DataByteSize;
end
% how much data on GPU
% things dependent of block size == size(data,3)  ==> thus total per : size(data,1)*size(data,2)*size(data,3
%     1. background,bg, phase, psf, data
%     == total -> 5
% things that are image size == size(data,1)*size(data,2)
%     1. aperture, ratio psf, ratio data, bp mask, denomObject,NumObject,
%     gk,hk,gk_freq,hk_freq,ratio_FT *2, temp, out psf, tv image, atan_tv,
%     objectEstimate, old_estimate, ratio mask
%     == total -> 23 (make 25 just incase)
RequiredMemory = RequiredMemory_blockDependent*5 + RequiredMemory_frameDependent*25; %total number of images 
 if (gpuID.AvailableMemory <=  RequiredMemory) % Memory in bytes
    % The input data is larger than the GPU's available memory, notify the
    % user that they do not have enough memory to process their input data
    errordlg('The input data requires more memory than the GPU currently has available, Please Cancel Deblur. Suggestion: Deblur using less frames/block or without Super Resolution.');
    return;
end
% push all inputs on gpuArray
data = gpuArray(single(data));
phase = gpuArray(single(phase));
bg = gpuArray(single(bg));
lambda_dn = gpuArray(single(processingParams.lambda));
shortenPhase = gpuArray(single(processingParams.shortenPhase));
aperture.A = gpuArray(single(aperture.A));
aperture.x = gpuArray(single(aperture.x));
physicalParams = structTransferGPU(physicalParams,0);
processingParams = structTransferGPU(processingParams,0);
aperture = structTransferGPU(aperture,0);
iterations = processingParams.iterations;


DeblurCancelled = 0;
superRes = processingParams.superRes;
disp = processingParams.disp;
[N1,N2,K] = size(data); % size of data (data is square)
background   =  gpuArray();

offset = 1e-3*median(data(:)); % epsilon set to 10^-6 of max value in data
data =fftshift(data(:,:,:)); %shift data to 4 corners
% resize the data, aperture, phase for super resolution
% make a 9 pixel blurring kernel, update N
if superRes 
    N1 = 3*N1;    N2 = 3*N2;
    dataOriginal = data;
    data = gpuArray(zeros(N1,N2,K,'single')); 
    data(2:3:end,2:3:end,:) = dataOriginal;
%   matlab limits num of images in array when doing imresize with GPU arrays to 50
    if K >50
       for i= 1:50:K
         try
         phase(:,:,i:i+50-1)= imresize(phase(:,:,i:i+50-1),[N1,N2]);
         catch % if at the end chuck
             phase(:,:,i:end)= imresize(phase(:,:,i:end),[N1,N2]);
         end
       end
    end
    aperture.A = imresize(aperture.A,[N1,N2]);
    aperture.x = linspace(aperture.x(1),aperture.x(end),N2);
    ghat = gpuArray(zeros(N1));
    ghat([1:2,N1],[1:2,N2]) = 1/9;
    ghat = fft2(ghat,N1,N2); % 9 pixel averaging kernel
end

if ~superRes
    Mask = processingParams.Mask;
    if var(var( fftshift(bg(:,:,1)))) ==0 % if background is all one value
         background = (bg(:,:,:)); 
    else
        background = fftshift(bg(:,:,:));
    end
else
    Mask = gpuArray(zeros(N1,N2));
    % Mask(2:3:end,2:3:end) = processingParams.Mask;
    if size(processingParams.Mask,1)==size(Mask,1)
        processingParams.Mask = processingParams.Mask(N1/3,N2/3);
    end
    Mask(2:3:end,2:3:end) = processingParams.Mask;
        
    background   =  gpuArray(zeros(N1,N2,K));
    if var(var( fftshift(bg(:,:,1)))) ==0  % if background is all one value
         bg_used = (bg);
    else
        bg_used = fftshift(bg);
    end
    background(2:3:end,2:3:end,:) = bg_used;
    background = real(ifft2(fft2(background).*ghat))*9;
end

phaseEstimate         = phase;     % first phase guess   -- gaussian
objectEstimate    = max(data(:,:,1)-background(:,:,1),1);
denomObject = K; % object estimate is divided by number of frames

%smooth input data for first guess
[~,gfft] = makeGausPSF([N1,N2],1);
objectEstimateFT = fft2(single(objectEstimate)).*gfft; % conv
% additional conv below for super resolution, and renormalization of denom
if superRes 
    denomObject = K/9; 
    objectEstimateFT = objectEstimateFT.*ghat.*ghat.*ghat.*9;
end
objectEstimate = real((ifft2(objectEstimateFT))); % updating object estimate

objectKnown  = processingParams.objectKnown; % flag indicating if object is known   
phaseKnown   = processingParams.phaseKnown;  % flag indicating if phase is known
prIterations = processingParams.prIterations;  % phase iteration -- inner loop       
tiltFlag = processingParams.tiltFlag;  % flag for tilt preservation
 
% PSF for each inputed object frame
for k = 1:K
  pointSpread(k) = makePSF(aperture.A, phaseEstimate(:,:,k), physicalParams);
  if superRes, pointSpread(k).H = pointSpread(k).H .* ghat; end
end

% variables need if displaying iteration information
if disp
    out.rel_chg_sol = gpuArray(single(zeros(iterations,1)));
    out.rel_chg_phase = gpuArray(single(zeros(iterations,K)));
    out.rel_chg_psf = gpuArray(single(zeros(iterations,K)));
    out.rel_chg_bg = gpuArray(single(zeros(iterations,K)));
    out.objF = gpuArray(single(zeros(iterations,1)));
    out.regF = out.objF;
    dataEstAll = gpuArray(single(zeros(N1,N2)));  % variable for finding bad pixels
end

% implement the mask for the bad pixels found from the previous block
if processingParams.badPixel
    % dataUsedOriginal = dataUsed; % save unmasked dataUsed
    % if sum(abs(diff(processingParams.Mask(:)))) %ignore if mask is all ones
    %    [MaskFT,denomObject,dataUsed] = initBadPixel(dataUsed,denomObject,pointSpread,processingParams);     
    % end   
    MaskFT = fft2(Mask);
    denomObject = zeros(N1,N2);
    for k = 1:K
        denomObject = denomObject + MaskFT.*conj(pointSpread(k).H);
    end
    denomObject = real(ifft2(denomObject));
end
if sum(Mask(:)) == 0, Mask = gpuArray(ones(size(objectEstimate))); end
    
TVoperator = getMHOTVfunctionDerGPU(processingParams.order,processingParams.levels,N1,N2);







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       MAIN LOOP BEGINS HERE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
objectEstimatep = objectEstimate; % save previous object estimates for acceleration
for ii = 1:iterations % while within tolerance or iteration bounds set by user
  if  disp, dataEstAll(:) = 0; end
  % acceleration: replace object estimate with this intermediate estimate
  % acceleration can be nullified by setting alpha coefficient to zero
  if processingParams.acceleration, NestAlpha = (ii-1)/(ii+2);
  else, NestAlpha = 0;
  end
  intermediateObject = max(objectEstimate +...
      NestAlpha*(objectEstimate-objectEstimatep),1e-4);
  
  if ~objectKnown  % if object unknown
     objectEstimateFT  = fft2(intermediateObject);       
     numerObject       = zeros(N1,N2);
  end
  % outer loop
  for k = 1:K  % for all the inputed images
      
    % evaluate the ratio of the data to the data estimate 
    if ~objectKnown        
       dataEstimate = abs(ifft2(objectEstimateFT .* pointSpread(k).H)).*Mask;
       ratio    = ((single(data(:,:,k)) + single(offset))...
           ./(dataEstimate + single(offset) + single(background(:,:,k))));
       ratioFT  = fft2(Mask.*ratio);
       
       % correlate the ratio with the point-spread function, and
       % accumulate the numerator for the object-estimate update            
       numerObject = numerObject + ratioFT .* conj(pointSpread(k).H); % take ifft2 later
       % add data estimates for all frames
       if disp
           dataEstAll = dataEstAll + dataEstimate + single(background(:,:,k));
       end
    end
    % update the psf estimate (inner loop)
    if ~phaseKnown
      if shortenPhase == 1 && flag == 0 && ii == 35
          prIterations = 1;
      end
      for prIter = 1:prIterations
          gk_freq        = aperture.A .* exp(1j*phaseEstimate(:,:,k)); %B.2
          gk             = ifft2(gk_freq);
          hk_temp        = abs(gk).^2; 
          hk             = hk_temp./(sum(sum(hk_temp))); % normlise hk since A is not 
          if ~superRes, dataEstimate   = abs(ifft2(objectEstimateFT .* fft2(hk)));
          else, dataEstimate = abs(ifft2(objectEstimateFT .* fft2(hk).*ghat));  
          end
          ratio          = (single(data(:,:,k)) + single(offset))./...
              ( single(dataEstimate) + single(offset) + (background(:,:,k))); %A.1
          ratioFT        = fft2(ratio);
          if ~superRes, numerPSFk = abs(ifft2(ratioFT .* conj(objectEstimateFT))); % A.2
          else, numerPSFk = abs(ifft2(ratioFT.*conj(objectEstimateFT).*conj(ghat)));
          end
          Gk             = fft2(gk .* numerPSFk); % A.4
          phaseEstimate(:,:,k) = atan2(imag(aperture.A.*Gk),real(aperture.A.*Gk));
          
      end
      pointSpread(k) = makePSF(aperture.A, phaseEstimate(:,:,k), physicalParams);
      if superRes, pointSpread(k).H = pointSpread(k).H .* ghat; end    
   end
  end
   
  % update the object estimate per iteration or tolarance
  objectEstimatep = objectEstimate;
  if ~objectKnown
      % ML update
      objectEstimate = (intermediateObject.*real((ifft2(numerObject))./denomObject));
      % total variation penalty (MAP update)
      tvm = -TVoperator(objectEstimate);
      atan_tv = (-atan(lambda_dn.*tvm)./(pi))+1;
      objectEstimate=real((objectEstimate)./(atan_tv));
  end
  diagnostic.object(ii) = sum(sum(fftshift(objectEstimate)));
  diagnostic.phase(ii) = sum(sum(sum(fftshift(phaseEstimate))));

  
 % Added as part of the multi-wait bar feature MFBD-33 
 % Displays the current iteration number that was ran on the current
%  try      
%      Mwb.Update(3, 1, gather(ii)/gather(processingParams.iterations),...
%          ['Block Iteration: ', num2str(ii),' of ',num2str(processingParams.iterations)]);
%      DeblurCancelled = 0;
%  catch 
%      % User cancelled
%      processingParams=-1;
%      out_psf=-1;
%      DeblurCancelled = 1;
%      diagnostic = 0;
%      reset(gpuID);
%      return;     
%  end

 rel_chg_sol = sum(abs(objectEstimatep(:)-objectEstimate(:)))/sum(abs(objectEstimatep(:)));
 if rel_chg_sol<processingParams.tol, break; end
 
 % an iterative display: one may add to any of these displays
 if disp
     out = myMFBDDisplay(out,objectEstimate,ii,data,dataEstAll...
         ,phaseEstimate,pointSpread,background,iterations,TVoperator);
 end            
  
  % denom object needs updating for bad pixels
  if processingParams.badPixel  
      % set bad pixels to zero and update denomObject as a normalization 
      denomObject(:) = 0;       
      for j = 1:K 
         denomObject = denomObject + MaskFT.*conj(pointSpread(j).H); 
      end 
      denomObject = real(ifft2(denomObject)); 
  end
end
% end plotting of estimte images ----------------------------------------

if processingParams.automateLambda
   % fast automated selection of lambda using the maximum evidence approach
   % this lambda is passed out and used on the next block
   lambdaParms.order = processingParams.order;
   lambdaParms.levels = processingParams.levels;
   lambdaParms.theta = .1;
   lambdaParms.tol = 1e-4;
   if superRes
       % in the case of super resolution, just hack the problem and
       % determine the parameter at the original resolution
       h = gpuArray(zeros(N1/3,N2/3,K));
       for i = 1:K
           tmp = makePSF(aperture.A, phaseEstimate(:,:,i), physicalParams);
           h(:,:,i) = imresize(tmp.psf,[N1/3,N2/3]);
           h(:,:,i) = h(:,:,i)/sum(sum(h(:,:,i)));
       end
       tmp = data - background;
       [~,MEout] = MaximumEvidenceLambda(h,tmp(2:3:end,2:3:end,:),lambdaParms); 
       tmp = data(2:3:end,2:3:end,:);
       processingParams.lambda = real(sqrt(2)*MEout.sigmas(end)/sqrt(MEout.etas(end))/median(tmp(:)));       
   else
        h = gpuArray(zeros(N1,N2,K));
        for i = 1:K, h(:,:,i) = pointSpread(i).psf; end
        [~,MEout] = MaximumEvidenceLambda(h,(data-background),lambdaParms); 
        processingParams.lambda = real(sqrt(2)*MEout.sigmas(end)/sqrt(MEout.etas(end))/median(data(:)));
   end
end

% update objective function to include contant terms so that it matches the
% true log-likelihood
if disp
    % I commented this out because it ruins things for super resolution
    out.logLikelihood = out.objF;%  + sum(data(:) - data(:).*log(data(:)));
    out.objF = real(out.objF - lambda_dn*out.regF);
end

% align the object with the last frame
if tiltFlag
    processingParams.objectEstimate =...
    local_cross_corr(ifftshift(data(:,:,1)),ifftshift(objectEstimate));
else
    processingParams.objectEstimate = ifftshift(objectEstimate);
end
processingParams.phaseEstimate  = phaseEstimate;
  
% sent back to cpu
processingParams = structTransferGPU(processingParams,1);
if disp
    processingParams.out = structTransferGPU(out,1);
end
%outputs
out_psf = gather(fftshift(pointSpread(K).psf));
diagnostic.object = gather(diagnostic.object);
diagnostic.phase = gather(diagnostic.phase);
% reset(gpuID)

%  try      
%      Mwb.Update(3, 1, 1,...
%          ['Block Iteration: ', num2str(processingParams.iterations),' of ',num2str(processingParams.iterations)]);
%      DeblurCancelled = 0;
%  catch 
%      % User cancelled
%      processingParams=-1;
%      out_psf=-1;
%      DeblurCancelled = 1;
%      return;     
%  end


end
     

function T = local_cross_corr(A,T)
    % cross-correlation of A and T
    cc = ifft2(fft2(A).*conj(fft2(T)));
    [~,b]=max(cc);
    [~,c]=max(max(cc));
    sh1 = b(c)-1;
    sh2 = c-1;
    T = circshift(T,[sh1,sh2]);
end