%% Appendix A.6: Flanger

% Import original audio
[x,fs] = audioread('noflange.wav');

% Delay value
D = round(0.003*fs);
% For input to cosine function
F = 2/fs;
% Internal delay buffer for x(n)
w = zeros(1, D + 1);
% Delay buffer index variable
q = 1;
% Amplitude
a = 0.9;

% Determine length of input signal
[N,k] = size(x);

% Loop through input signal
for n = 1:N
    d = round((D/2)*(1 - cos(2*pi*F*n)));
    tap = q + d;
    if tap < 1
        tap = tap + (D + 1);
    end
    if tap > (D + 1)
        tap = tap - (D + 1);
    end
    y(n) = x(n) + a*w(tap);
    w(q) = x(n);
    q = q - 1;
    if q < 1
        q = D + 1;
    end
end

% Normalize y(n)
ymax = max(y);
y = y/ymax;

% Listen to results
sound(y,fs);

%% Plot Results
t = 1:N;
subplot(2,1,1)
plot(t/fs,x),title('no flange')
subplot(2,1,2)
plot(t/fs,y); title('flanged'), xlabel('t (sec)');

%% Output Results
audiowrite('FlangedFile.wav',y,fs);