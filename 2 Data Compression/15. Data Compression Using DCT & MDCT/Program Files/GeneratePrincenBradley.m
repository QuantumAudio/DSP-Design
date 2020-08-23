%% Generating Princen-Bradley Windows
N = 4; beta = 5;
rectangular = pbwin(N,0,beta)
sine = pbwin(N,1,beta)
vorbis = pbwin(N,2,beta)
KBD = pbwin(N,3,beta)
%% Verifying Princen-Bradley Condition
rectangular(1:N).^2 + rectangular(N+1:2*N).^2
sine(1:N).^2 + sine(N+1:2*N).^2
vorbis(1:N).^2 + vorbis(N+1:2*N).^2
KBD(1:N).^2 + KBD(N+1:2*N).^2
%% Plot Results
N = 128, beta = 20;
n = 0:2*N-1;
close all
plot(n,sine,'r',n,vorbis,'g',n,KBD,'b'); axis([0,256,0,1.5]);
legend('sine','vorbis','KBD'); grid on; xlabel('n')
title('Princen-Bradley windows, 2N = 256, \beta = 20');