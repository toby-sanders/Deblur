clear;
d = 128;
P = .1;

[psf,psf1,psf2] = makeLangPSF(d,P);

figure(212);hold off;
plot(psf);hold on;
% plot(psf1);
% plot(psf2);
% hold off;