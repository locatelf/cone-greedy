function [x,errors] = nnomp_canon(A,y,IT)
% Canonical Implementation of Non-Negative Orthogonal Matching Pursuit with a real value dictionary
% Important: this implementation guarantees the positivity of the
% coefficients by solving nnls problem in each iteration!

% Copyright (c) 2014, Mehrdad Yaghoobi 
% University of Edinburgh, 

r = y;
[m,n] = size(A);
x = zeros(n,1);
k = 1;
mag = 1;
s = [];
bpr = A'*r;
xs = [];
err = 1e-10;

part = zeros(n,1);
errors = zeros(IT,1);
while k <= IT %&& mag>0
    bpr(s) = 0;
    [mag,ind] = max(bpr);    
    if mag > 0       
        s = [s;ind];
        As = A(:,s);
%         xs = nnls(As, y, err); 
        xs = lsqnonneg(As,y);
        r = y - As*xs;
        bpr = A'*r;
    end        
   
    
    part(s) = xs;
    errors(k)=norm(y-A*part)^2;
    k = k+1;
end
k = k-1;
% x(s) = R(1:k,1:k)\z(1:k)';
x(s) = xs;
