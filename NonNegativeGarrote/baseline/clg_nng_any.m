function [beta,beta0,loglik]=clg_nng_any(X,y,bstart,bstart0,lambda_nng)
%CLG_NNG_ANY Non-negative garotte logistic regression with user-supplied initial estimates
%  [beta, beta0, loglik, bridge, bridge0] = clg_nng(X, y, bstart, bstart0, lambda_nng)
%
%   The input arguments are:
%       X            - [n x p] data matrix (without the intercept)
%       y            - [n x 1] target vector, coded as 0 and 1
%       bstart       - [p x 1] vector of initial estimates
%       bstart0      - [1 x 1] initial estimate of intercept
%       lambda_nng   - [1 x NLAMBDA] vector of NNG penalties to try (0 = no penalisation)
%
%   Return values:
%       beta        - [p x NLAMBDA] NNG estimates for each value of lambda_nng
%       beta0       - [1 x NLAMBDA] estimates of intercept for each value of lambda_nng
%       loglik      - log-likelihood of the NNG estimates for each value of lambda_nng
%
%   Please see examples_bayesreg.m for usage examples.
%
%   To cite this code:
%     Makalic, E. & Schmidt, D. F.
%     Logistic Regression with the Non-Negative Garotte
%     Lecture Notes in Computer Science, Vol. 7106, pp. 82-91, 2011
%
%   References:
%     Breiman, L. 
%     Better subset regression using the nonnegative garrote
%     Technometrics , Vol. 37, pp. 373-384, 1995
%
%     Genkin, A., Lewis, D.D. & Madigan, D
%     Large-scale Bayesian logistic regression for text categorization 
%     Technometrics, Vol. 49, pp. 291-304, 2007
%
%   (c) Copyright Enes Makalic and Daniel F. Schmidt, 2011

%% data details
[n,d] = size(X);

%% Prepare data
% Convert y's from (0,1) to (-1,1)
U = sort(unique(y))';
if (length(U) ~= 2 || ~all(U == [0,1]))
    error('Targets must be coded as 0 and 1.');
end
y(y == 0) = -1;

% check starting points
if(~isvector(bstart) || length(bstart) ~= d)
    error('Starting estimates must be a [%d x 1] vector.', d);
end
if(~isscalar(bstart0))
    error('Starting intercept must be a scalar.');
end

% resize vector
if (~isvector(lambda_nng))
    error('Lambda values for NNG must be a vector.');
end
[a,~] = size(lambda_nng);
if (a == 1)
    lambda_nng = lambda_nng';
end

% check if an intercept was passed -- if so, generate error
S = sum(X - ones(n,d), 1);
if (any(S == 0))
    error('X matrix should not include an intercept column');
end

% now add in the intercept -- always the first column
X = [ones(n,1), X];

%% find r
bstart = [bstart0; bstart];
r = y.*(X*bstart);

% modify X for NNG
X(:,2:end)=bsxfun(@times,X(:,2:end),bstart(2:end)');

beta = [bstart(1); ones(d,1)];    % starting estimates (now that X is modified)
                                  % this is not 'really' beta, it's c where
                                  % beta_nng = c * beta_rr

% do LASSO on the new X and starting at beta
% LASSO is the same as NNG now as c_i > 0 (i=1,...,d)
[beta,loglik] = clg_lasso_mod(X,y,lambda_nng,beta,r);                                

% modify beta to 'work' on the original X scale
beta0 = beta(1,:);
beta = bsxfun(@times,beta(2:end,:),bstart(2:end));

%% done
return;