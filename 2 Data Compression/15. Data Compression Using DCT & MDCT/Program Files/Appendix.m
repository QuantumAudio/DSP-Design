%% Appendix A.1: DCT Basis Functions

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

%% Appendix A.2: dctf Function
function D = dctf(X)
% forward DCT of the matrix X

[N,M] = size(X);

Y = [X; flipud(X)]; % Construct y column vectors 

Yk = fft(Y);    % Take fft of each column

% Define weighting coefficients of DFT
k = (0:2*N-1)';                     % Vectorize DFT column index k
weights = (1/2)*exp(-j*pi*k/(2*N)); % Repeat for M columns 
WEIGHTS = repmat(weights,1,M); 
% Calculate C_ext as DFT of Yk
Ck_ext = WEIGHTS.*Yk;                        
% Keep first N elements of each column
Ck = Ck_ext(1:N,:);    

% Construct normalizing s vector
s0 = sqrt(N);   
sk = sqrt(N/2)*ones(N-1,1);
s = [s0;sk];
% Repeat for M columns
S = repmat(s,1,M);
            
% Normalize output matrix of DCT coefficients
D = real(Ck./S);

end

%% Appendix A.3: idct Function
function X = idctf(D)
% Computes the IDCT of D

[N,M] = size(D);    % column and row lengths

k = (0:2*N-1)';    % Vectorize Ck_ext index k

% Construct normalizing, s vector
s0 = sqrt(N);   
sk = sqrt(N/2)*ones(N-1,1);
s = [s0;sk];

S = repmat(s,1,M);  % Scale D to Ck
Ck = S.*D;

Ck_ext = [Ck; zeros(1,M); -flipud(Ck(2:N,:))]; % Extend Ck 

% Reconstruct Yk from Ck_ext
weights = 2*exp(j*pi.*k/(2*N));                 
WEIGHTS = repmat(weights,1,M);

Yk = WEIGHTS.*Ck_ext; 

Y = real(ifft(Yk));     % Take the inverse fft

X = Y(1:N,:);           % Keep the first N values of each column

end

%% Appendix A.4: Comparing DCT Implementations

% Normalized DCT Matrix A for arbitray value of N
A = @(n) sqrt(2/N)*[ones(1,N)/sqrt(2); cos(pi*(1:N-1)'*(((0:N-1) + 1/2)/N))];
% Define a random matrix X
X = rand(1024,100);

% Built-in dct Implementation
tic
D = dct(X);
Xdct = idct(D);
t_dct = toc;

% dctf Implementation
tic 
D = dctf(X);    % forward DCT
Xdctf = idctf(D);
t_dctf = toc;

% Matrix Implementation
tic
N = 1024;
n = 1:N;
A_dct = A(n);
D = A_dct*X;
Xmat = A_dct'*D;
t_matrix = toc;

% Comparing the Different Implementations
format long
disp('X = ');
disp(X(1:10,1:3));
disp('Xdct = ');
disp(Xdct(1:10,1:3));
disp('Xdctf = ');
disp(Xdctf(1:10,1:3));
disp('Xmat = ');
disp(Xmat(1:10,1:3));

XdctErr = abs(X - Xdct);
disp('XdctError = ');
disp(XdctErr(1:10,1:3));

XdctfErr = abs(X - Xdctf);
disp('XdctfError = ');
disp(XdctfErr(1:10,1:3));

XmatErr = abs(X - Xmat);
disp('XmatError = ');
disp(XmatErr(1:10,1:3));

% t_dct
% t_dctf
% t_matrix

%% Appendix A.5: function [y, ra] = dctcompr(x, N, r, method);
% Implements the DCT compression on input signal x using 1 of 2 methods.
% If method = 1, a fixed number of values are kept per column
% If method = 2, values above a certain threshold are kept.
% N = number of input samples per block
% r = compression ration (between 0 and 1)

X = buffer(x,N);
D = dct(X);
M = size(D,2);

if method == 1          % Keep a fixed number of values per column
    Nr = round(r*N);     % Number of values kept in each frame            
    ra = Nr/N;          % actual compression ratio
    [~,Ir] = sort(abs(D), 'descend'); % Sort columns descending
    Ir = Ir(1:Nr,:);                   % Keep highest Nr values
    C = zeros(size(D));                 % kept DCT coefficients
    for m = 1:M
        Dr(:,m) = D(Ir(:,m),m);         % Nr x M matrix of sorted coeficients
        C(Ir(:,m),m) = Dr(:,m);         % C = new DCT coefficients
    end
end
   
if method == 2          % Keep values above a certain threshold
    Dthr = r*max(max(abs(D)));          % ratio times the max value in D
    I = find(abs(D) < Dthr);            % Indices of coefficients to be discarded
    C = D;                              % Copy of D
    C(I) = 0;                           % Discard coefficients below Dthr
    ra = 1 - length(I)/(N*M);           % Actual compression ratio
end
    
Y = idct(C);              % Take inverse dct of kept coefficients
y = Y(:);                 % Concatenate columns
y = y(1:length(x));       % Make output the same length as the input

end

%% Appendix A.6: DCT Compression Examples

% Example Input Signal
L = 40; t = (0:L-1)/L;
x = sin(10*t.^2) + 2*t;
N = 10;

% Method-1, r = 0.4, r_actual = 0.4
method = 1;
r = 0.4;
[y, ra] = dctcompr(x,N,r,1);
close all
plot(t,x,'r-',t,y,'b.'); title('method-1, N = 10, r = 0.4')
xlabel('t'),legend('original','recovered')

% Method-2, r = 0.014, r_actual = 0.4
method = 2;
r = 0.014;
[y, ra] = dctcompr(x,N,r,2);
close all
plot(t,x,'r-',t,y,'b.'); title('method-2, N = 10, r_{thr} = 0.014')
xlabel('t'),legend('original','recovered')

% dctcompr, method-1 on audio signal
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

% 2D-DCT on an image file
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

%% Appendix A.7 pbwin Function
function w = pbwin(N, type, beta)
% Generates Princen-Bradley windows
%
% type = 0 generates a rectangular window
% type = 1 generates a sine window
% type = 2 generates a vorbis window
% type = 3 generates a KBD window
% N = DCT block length
% beta = Kaiser shape parameter (for KBD window only)

n = 0:2*N-1;

if type == 0
    w = ones([1,2*N]);    % Generate a rectangular window
end

if type == 1
    w = sqrt(2)*sin((pi/(2*N))*(n + 0.5)); % Generate sine window
end

if type == 2
    w = sqrt(2)*sin((pi/2)*sin((pi/(2*N))*(n + 0.5)).^2); % Generate vorbis window
end


if type == 3
    k = 0:N;
    % Compute zeroth order Bessel function of the first kind
    fk = besseli(0, beta*sqrt(1-((k - N/2)/(N/2)).^2)); 

    S = sum(fk);

    for n = 0:N-1
        wf(n+1) = sqrt((2/S)*sum(fk(1:(n+1))));    % Compute forward half of window
    end
    
    w = [wf, fliplr(wf)];                       % Concatenate forward and reverse halves
end

end

%% Appendix A.8: Generating Princen-Bradly Windows
N = 4; beta = 5;
rectangular = pbwin(N,0,beta)
sine = pbwin(N,1,beta)
vorbis = pbwin(N,2,beta)
KBD = pbwin(N,3,beta)

% Verifying Princen-Bradley Condition
rectangular(1:N).^2 + rectangular(N+1:2*N).^2
sine(1:N).^2 + sine(N+1:2*N).^2
vorbis(1:N).^2 + vorbis(N+1:2*N).^2
KBD(1:N).^2 + KBD(N+1:2*N).^2

% Plot Results
N = 128, beta = 20;
n = 0:2*N-1;
close all
plot(n,sine,'r',n,vorbis,'g',n,KBD,'b'); axis([0,256,0,1.5]);
legend('sine','vorbis','KBD'); grid on; xlabel('n')
title('Princen-Bradley windows, 2N = 256, \beta = 20');

%% Appendix A.9: mdct Function
function D = mdct(X)
% forward MDCT of the matrix X

% Determine block size and shape
N = size(X,1)/2;
k = (0:N-1)';
n = 0:2*N-1;

% Determine coefficients
F = cos((pi/N*(k + 0.5))*(n + 0.5 + N/2));
D = F*X;

end 

%% Appendix A.10: imdct Function
function X = imdct(D)
% Inverse MDCT of the matrix D

% Determine block size and shape
N = size(D,1);
k = (0:N-1)';
n = 0:2*N-1;

% Compute coefficients
F = cos((pi/N*(k + 0.5))*(n + 0.5 + N/2));
X = (1/N)*F'*D;

end

%% Appendix A.11: mdctcompr Function
function [y, ra] = mdctcompr(x, N, rth, win, beta)
% Applied MDCT compression to the input signal x
% N =   DCT block length
% rth = threshold value
% win and beta are inputs to the pbwin function, which 
% generates Princen-Bradley windows
% win = 0 generates a rectangular window
% win = 1 generates a sine window
% win = 2 generates a vorbis window
% win = 3 generates a KBD window
% beta = Kaiser shape parameter (for KBD window only)
% 
X = buffer(x, 2*N, N, 'nodelay');
M = size(X,2);
w = (pbwin(N, win, beta))';      
W = repmat(w,1,M);
D = mdct(W.*X);

% DCT Method-2
Dthr = rth*max(max(abs(D)));          % ratio times the max value in D
I = find(abs(D) < Dthr);            % Indices of coefficients to be discarded
C = D;                              % Copy of D
C(I) = 0;                           % Discard coefficients below Dthr
ra = 1 - length(I)/(N*M);           % Actual compression ratio

Y = W.*imdct(C);

% Overlap-add function adjusted for blocks of length 2N
N = size(Y,1)/2;    % Y has columns of length 2*N
M = size(Y,2);      % M columns
L = N*M + N;                    % Extended length of y
y = zeros(L,1);                  % Initialize output vector

n = (1:2*N)';                   % Vectorize columns

for m = 0:M-1                     % Number of blocks (columns)         
    y(m*N + n) = y(m*N + n) + Y(:,m+1);  % add vectors with overlap of N = 50%
end
% End of adjusted overlap-add function

y = y(1:length(x));     % Make output length the same as the input length

end

%% Appendix A.12: TDAC Illustration
x = (1:28)';    % Example Input Signal
N = 4;          % overlap
beta = 6;       % KBD Parameter
win = 3;        % KBD window


X = buffer(x, 2*N, N, 'nodelay');
M = size(X,2);
w = (pbwin(N, win, beta))';      
W = repmat(w,1,M);
D = mdct(W.*X);
Y = W.*imdct(D);

% generate values to print table
y1 = [Y(:,1); zeros(20,1)];
y2 = [zeros(4,1);Y(:,2);zeros(16,1)];
y3 = [zeros(8,1);Y(:,3);zeros(12,1)];
y4 = [zeros(12,1);Y(:,4);zeros(8,1)];
y5 = [zeros(16,1);Y(:,5);zeros(4,1)];
y6 = [zeros(20,1);Y(:,6)];
y = y1 + y2 + y3 + y4 + y5 + y6;


fprintf('   x     y1       y2       y3       y4       y5       y6       y\n');
fprintf('--------------------------------------------------------------------\n');
fprintf('%3.0f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f \n',[x';y1';y2';y3';y4';y5';y6';y']);


%% Appendix A.13: MDCT Compression Examples

% Adusting rth until ra is in the range [0.15, 0.20] 
[x, fs] = audioread('flute2.wav'); % 4 sec audio sample
win = 1;
beta = 5;
rth = 0.0003;
N = 1024;
[y, ra] = mdctcompr(x, N, rth, win, beta);

% rth = 0.005
t = 0:0.01:0.99;    % 100 time instants in the interval [0,1)
x = sin(10*t.^2) + 2*t;        % signal samples
N = 20; rth = 0.005;
win = 3; beta = 15;
[y, ra] = mdctcompr(x, N, rth, win, beta);
close all
plot(t,x,'r-',t,y,'b'); title('2N = 40, r_{thr} 0.005, r_a = 0.225')
xlabel('t'), legend('original', 'compressed'), grid on

% rth = 0.05
t = 0:0.01:0.99;    % 100 time instants in the interval [0,1)
x = sin(10*t.^2) + 2*t;        % signal samples
N = 20; rth = 0.05;
win = 3; beta = 15;
[y, ra] = mdctcompr(x, N, rth, win, beta);
close all
plot(t,x,'r-',t,y,'b'); title('2N = 40, r_{thr} 0.050, r_a = 0.150')
xlabel('t'), legend('original', 'compressed'), grid on

