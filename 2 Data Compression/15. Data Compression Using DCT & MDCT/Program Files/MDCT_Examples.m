% Adusting rth until ra is in the range [0.15, 0.20] 
[x, fs] = audioread('flute2.wav'); % 4 sec audio sample
win = 1;
beta = 5;
rth = 0.0003;
N = 1024;
[y, ra] = mdctcompr(x, N, rth, win, beta);

%% rth = 0.005
t = 0:0.01:0.99;    % 100 time instants in the interval [0,1)
x = sin(10*t.^2) + 2*t;        % signal samples
N = 20; rth = 0.005;
win = 3; beta = 15;
[y, ra] = mdctcompr(x, N, rth, win, beta);
close all
plot(t,x,'r-',t,y,'b'); title('2N = 40, r_{thr} 0.005, r_a = 0.225')
xlabel('t'), legend('original', 'compressed'), grid on

%% rth = 0.05
t = 0:0.01:0.99;    % 100 time instants in the interval [0,1)
x = sin(10*t.^2) + 2*t;        % signal samples
N = 20; rth = 0.05;
win = 3; beta = 15;
[y, ra] = mdctcompr(x, N, rth, win, beta);
close all
plot(t,x,'r-',t,y,'b'); title('2N = 40, r_{thr} 0.050, r_a = 0.150')
xlabel('t'), legend('original', 'compressed'), grid on