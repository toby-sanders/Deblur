 clear;
 d = 320;

 h = randn(d,1);
 b = randn(d,1);

 H = zeros(d);
 for i = 1:d
     H(i,:) = circshift(flipud(h),i);
 end

 v1 = ifft2(fft2(h).*fft2(b));
 v2 = H*b;
v = 0:d-1;
F = exp(1i*2*pi/d*v'*v)/sqrt(d);
% figure(123);imagesc(real(F'*F));colorbar;
 myrel(v1,v2)


 trace1 = sum(H(1:d+1:end))
 trace2 = sum(fft(h))
 trace3 = h(1)*d
 trace0 = sum(col(F'*H*F))

% this shows how to compute the trace terms in the Poisson-SURE case
 trace4 = sum(col(H(1:d+1:end)).*b(:))
 trace5 = sum(fft2(h))/d*sum(b(:))

