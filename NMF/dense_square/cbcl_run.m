rng('default')
clear
close all
load('../data/cbcl.mat'); % For CBCL 
%load('reuters'); % For reuters corpus
%[V,~] = getdcm(0); % For XNIX
n = size(V,1);
m = size(V,2);
rankToTest = [10];
K = numel(rankToTest);
errGCD = zeros(K,1);
errNNALS = zeros(K,1);
errMult = zeros(K,1);

ik = 1;
for k = rankToTest
    W = abs(rand(n,k));
    H = abs(rand(k,m));
    max_iter = 100;
     [W0, H0, objGCD, timeGCD] = NMF_GCD(V,k,max_iter,W,H,1);
     errGCD(ik) = objGCD(end);
    [W2,H2] = nnmf(V,k,'replicates',10,'algorithm','als');
    N = W2*H2;
    errNNALS(ik) = 0.5*norm(V - N,'fro')^2;
    [W2,H2] = nnmf(V,k,'replicates',10,'algorithm','mult');
    N = W2*H2;
    errMult(ik) = 0.5*norm(V - N,'fro')^2;
    ik = ik + 1;
end

MaxR = max(rankToTest);
X0 = zeros(size(V));
[X_pred, residual,test_errorFC] = FCNMP(V,X0,MaxR,0,100,V);
[X_pred, residual,test_errorPW] = PWNMP(V,X0,MaxR,0,100,V);
errGCD
errNNALS
errMult
test_errorFC(rankToTest)
test_errorPW(rankToTest)

