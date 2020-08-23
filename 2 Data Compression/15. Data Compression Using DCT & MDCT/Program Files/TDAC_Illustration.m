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
