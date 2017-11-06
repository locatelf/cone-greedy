%clear, clc
rng('default')
path = 'sonar.csv';
close all
warning('off','all')
X = load(path);
y = X(:,end);
y(y==-1)=0;

X = normc(X(:,1:end-1)')';





n = size(X,1);
p = size(X,2);
Lval = 1000;
Lacc = zeros(1,numel(Lval));
cnt = 0;
for L=Lval
    cnt = cnt +1;
%% train test split
maxOutIt = 100;
errS = zeros(1,maxOutIt);
errP = zeros(1,maxOutIt);
errF = zeros(1,maxOutIt);
errCCD = zeros(1,maxOutIt);

trS = zeros(1,maxOutIt);
trP = zeros(1,maxOutIt);
trF = zeros(1,maxOutIt);
trCCD = zeros(1,maxOutIt);
sparsity = zeros(maxOutIt,4);
for outIt = 1:maxOutIt
outIt
rp = randperm(n);
prcTrn = 0.7;
nTrn = round(n*prcTrn);
idx = 1:n;
idxTrn = idx(rp(1:nTrn));
idxTst = idx(rp(nTrn+1:end));

XTrn = X(idxTrn,:);
yTrn = y(idxTrn);
XTst = X(idxTst,:);
yTst = y(idxTst);
addpath('baseline');
[model, llh] = logitBin(XTrn',yTrn');
 betaOld = model.w(1:end-1);
 bias = model.w(end);
[beta,beta0,loglik]=clg_nng_any(XTrn,yTrn,betaOld,bias,1);
sparsity(outIt,1)= numel(find(beta==0));
[errS(outIt),errP(outIt),errF(outIt),trS(outIt),trP(outIt),trF(outIt),sparsity(outIt,2:end)]=nngarrote(XTrn,yTrn,nTrn,p,XTst,yTst,L);
trCCD(outIt) = numel(find(yTrn-round(sigmoid(XTrn*beta+beta0))==0))/numel(yTrn);
errCCD(outIt)= numel(find(yTst-round(sigmoid(XTst*beta+beta0))==0))/numel(yTst);
end
fprintf('test')
 mean(errS)
 std(errS)
 mean(errP)
 std(errP)
 mean(errF)
  std(errF)
mean(errCCD)
std(errCCD)

fprintf('train')
 median(trS)
 std(trS)
 median(trP)
 std(trP)
 median(trF)
 std(trF)
median(trCCD)
std(trCCD)

mean(sparsity,1)
end
