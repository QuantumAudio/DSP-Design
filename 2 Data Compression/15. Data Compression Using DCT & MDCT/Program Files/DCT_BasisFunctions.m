% DCT Basis Functions
% number of basis functions
% N = 8
N = 16;
% for plotting
k = 0:N-1;
n = k';
% basis function parameters
s0 = sqrt(N);   
sk = sqrt(N/2)*ones(N-1,1);
s = [s0;sk];
% generate basis functions
Fk = (1./s).*cos((pi*k/N).*(n + 0.5)); % Each column of Fk is a basis function of n

% Plot basis functions
for l = 1:N
subplot(6,4,l)
stairs(Fk(:,l));
xlim([0,N]);
end
