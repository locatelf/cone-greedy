function [errS,errP,errF,trS,trP,trF,sparsity]=nngarrote(XTrn,yTrn,n,p,XTst,yTst,L)
 [model, llh] = logitBin(XTrn',yTrn');
 beta = model.w(1:end-1);
 bias = model.w(end);
% [B,FitInfo] = lasso(XTrn,yTrn,'lambda',0.0001);
% beta= B;
% bias = FitInfo.Intercept;
sparsity = zeros(1,3);
% ypred =XTst*beta+bias;
% ypred(ypred>0.5)=1;
% ypred(ypred<0.5)=0;
% yTst(yTst<0) = 0;
% numel(find(yTst-ypred==0))/numel(ypred)
lambda = 1;

% c = CCD(yTrn,XTrn,beta,n,p,lambda); 
% m.w = [beta.*c;bias];
% errCCD = numel(find(yTst'-logitBinPred(m,XTst')==0))/numel(yTst);
% trCCD = numel(find(yTrn'-logitBinPred(m,XTrn')==0))/numel(yTrn);

c = NNMPslow(yTrn,XTrn,beta,bias,n,p,lambda,L); 
m.w = [beta.*c;bias];
errS = numel(find(yTst'-logitBinPred(m,XTst')==0))/numel(yTst);
trS = numel(find(yTrn'-logitBinPred(m,XTrn')==0))/numel(yTrn);
sparsity(1)= numel(find(m.w==0));
c = PWNMP(yTrn,XTrn,beta,bias,n,p,lambda,L); 
m.w = [beta.*c;bias];
errP =numel(find(yTst'-logitBinPred(m,XTst')==0))/numel(yTst);
trP = numel(find(yTrn'-logitBinPred(m,XTrn')==0))/numel(yTrn);
sparsity(2)= numel(find(m.w==0));

c = FCNMP(yTrn,XTrn,beta,bias,n,p,lambda,L); 
m.w = [beta.*c;bias];
errF = numel(find(yTst'-logitBinPred(m,XTst')==0))/numel(yTst);
trF = numel(find(yTrn'-logitBinPred(m,XTrn')==0))/numel(yTrn);
sparsity(3)= numel(find(m.w==0));

end