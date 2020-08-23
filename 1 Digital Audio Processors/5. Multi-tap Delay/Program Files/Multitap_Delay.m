%% Appendix A.5: Multi-tap Delay

% Import original audio
[s,fs] = audioread('original.wav');
% Delay values
D1 = 0.125*fs; D2 = 0.25*fs;
% Amplitude coefficients
b0 = 1; b1 = 1; b2 = 1;
a1 = 0.2; a2 = 0.4;

% Determine length of input signal
[N,k] = size(s);

% Internal delay buffer for s(n)
w = zeros(1, D1 + D2 + 1);
% Delay buffer index variable
q = 1;
% Delay buffer taps
tap1 = D1 + 1;
tap2 = D1 + D2 + 1;
% Loop through input signal
tic
for n = 1:N
    s1 = w(tap1);
    s2 = w(tap2);
    y(n) = b0*s(n) + b1*s1 + b2*s2;
    w(q) = s(n) + a1*s1 + a2*s2;
        q = q - 1;                 % Backshift index 1
    if q < 1                    % Circulate index 1
        q = D1 + D2 + 1;
    end
    tap1 = tap1 - 1;             % Backshift tap1
    if tap1 < 1                  % Circulate tap1
        tap1 = D1 + D2 + 1;
    end
        tap2 = tap2 - 1;         % Backshift tap2
    if tap2 < 1                  % Circulate tap2
        tap2 = D1 + D2 + 1;
    end
end
toc
% Normalize y(n)
ymax = max(y);
y = y/ymax;

% Check results
sound(y,fs);

%% Compare to filter Function
n = [b0, zeros(1,D1 - 1), b1-a1*b0, zeros(1,D2 - 1),b2 - a2*b0];
d = [1, zeros(1,D1 - 1), -a1, zeros(1,D2 - 1),-a2];

tic
yfilt = filter(n,d,s);
toc

% Normalize yfilt(n)
yfiltmax = max(yfilt);
yfilt = yfilt/yfiltmax;
sound(yfilt,fs);

%% Plot Results
t = 1:160000;
subplot(2,1,1)
plot(t/fs,s),title('original')
subplot(2,1,2)
plot(t/fs,y); title('multitap'), xlabel('t (sec)');

%% Output Results
audiowrite('Multi-Tap.wav',y,fs)