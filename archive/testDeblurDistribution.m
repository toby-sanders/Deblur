clear;
m = 1000;
n = 512+88;


R = getDeblur_distribution(m,n);

I = zeros(m,n);
for i = 1:size(R,2)
    for j = 1:size(R,3)
        indX = R(3,i,j):R(4,i,j);
        indY = R(1,i,j):R(2,i,j);
        I(indY,indX) = I(indY,indX) + 1;
    end
end


figure(4123);imagesc(I);
colorbar;