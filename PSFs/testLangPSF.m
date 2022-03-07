clear;
d = 64;
P = .25;

[psf] = makeLangPSF(d,P);

figure(212);hold off;
plot(psf);hold on;
%  plot(psf0);
% plot(diff(psf)/P);
% plot(psf2);
% hold off;