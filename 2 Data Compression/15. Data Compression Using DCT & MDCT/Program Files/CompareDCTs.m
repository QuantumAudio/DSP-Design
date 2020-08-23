% Normalized DCT Matrix A for arbitray value of N
A = @(n) sqrt(2/N)*[ones(1,N)/sqrt(2); cos(pi*(1:N-1)'*(((0:N-1) + 1/2)/N))];
% Define a random matrix X
X = rand(1024,100);

%% Built-in dct Implementation
tic
D = dct(X);
Xdct = idct(D);
t_dct = toc;

%% dctf Implementation
tic 
D = dctf(X);    % forward DCT
Xdctf = idctf(D);
t_dctf = toc;

%% Matrix Implementation
tic
N = 1024;
n = 1:N;
A_dct = A(n);
D = A_dct*X;
Xmat = A_dct'*D;
t_matrix = toc;

%% Comparing the Different Implementations
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
