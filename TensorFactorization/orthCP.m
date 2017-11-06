% Generate CPD, where one factor has orthonormal columns.
rng('default')

X=getdcm();
n = numel(X);
idx = 1:n;
prc_trn = 0.1;
n_trn = round(n*prc_trn);
rp = randperm(n);
idx_trn = idx(rp(1:n_trn));
idx_tst = idx(rp(n_trn+1:end));

X_trn = zeros(size(X));
X_trn(idx_trn) = X(idx_trn);

X_tst = zeros(size(X));
X_tst(idx_tst) = X(idx_tst);

X0 = rand(size(X_trn));
low_rank = 10;

R = 10;
I1=512;
I2=512;
I3=24;

% Clear previous model
model = struct;
% Define model variables.
model.variables.q = randn(I1*R-0.5*R*(R-1),1);
model.variables.b = randn(I2*R-0.5*R*(R-1),1,1);
model.variables.c = randn(I3*R-0.5*R*(R-1),1,1);
% Define a function that transforms q into Q.
% This anonymous function stores the size of Q as its last argument. 
orthq = @(z,task)struct_orth(z,task,[I1 R]);
orthb = @(z,task)struct_orth(z,task,[I2 R]);
orthc = @(z,task)struct_orth(z,task,[I3 R]);

% Define model factors.
model.factors.Q = { 'q' ,orthq}; % Create Q as orthq(q). 
model.factors.B =  {'b', orthb} ;
model.factors.C =  {'c', orthc} ;

% Define model factorizations.
model.factorizations.myfac.data = X_trn;
model.factorizations.myfac.cpd  = { 'Q' , 'B' , 'C' };

tic
X_trn(X_trn==0) = NaN;
X_hat = sdf_nls(model,  'Display' , 100);%cpdgen(cpd(X_trn,low_rank));
X_hat=cpdgen({X_hat.factors.Q,X_hat.factors.B,X_hat.factors.C});
elapsedTime = toc;
rmse_2 = sqrt(mean((X_tst(X_tst ~= nil) - X_hat(X_tst ~= nil)).^2));  % error on known test values
fprintf('Elapsed time: %.2fs\n', elapsedTime)
disp(['Root of Mean-squared error on test set: ' num2str(rmse_2)]);
figure
imshow(X_hat(:,:,1), []);
