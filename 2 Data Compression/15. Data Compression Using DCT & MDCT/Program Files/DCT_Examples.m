% Example Input Signal
L = 40; t = (0:L-1)/L;
x = sin(10*t.^2) + 2*t;
N = 10;

%% Method-1, r = 0.4, r_actual = 0.4
method = 1;
r = 0.4;
[y, ra] = dctcompr(x,N,r,1);
close all
plot(t,x,'r-',t,y,'b.'); title('method-1, N = 10, r = 0.4')
xlabel('t'),legend('original','recovered')

%% Method-2, r = 0.014, r_actual = 0.4
method = 2;
r = 0.014;
[y, ra] = dctcompr(x,N,r,2);
close all
plot(t,x,'r-',t,y,'b.'); title('method-2, N = 10, r_{thr} = 0.014')
xlabel('t'),legend('original','recovered')

%% dctcompr, method-1 on audio signal
[x, fs] = audioread('flute2.wav'); % 4 sec audio sample
x = x(:,1); % as a column
method = 1;
r = 0.2;
[y, ra] = dctcompr(x,N,r,1);
y = y/max(abs(y));
% Listen to results
soundsc(y,fs);
% Output results
audiowrite('dctcompression.wav', [x', zeros(1,fs),y'],fs);

%% 2D-DCT on an image file
X = imread('cameraman.tif');    % read image, 256x256 matrix
D = dct2(X);                    % compute its 2D-DCT

Dmax = max(max(abs(D)));        % Dmax = 30393.4687
Dth = 10;                       % select a threshold
rth = Dth/Dmax;                 % with threshold factor, rth = 3.2902e-04
C = D;
C(abs(D)<Dth) = 0;              % compressed DCT
ra = length(find(C))/prod(size(C)); % actual compression ratio,  ra = 26617/65536 = 0.4061
Y = idct2(C);                       % inverse 2D-DCT
figure; imshowpair(X,Y,'montage')   % display images side by side