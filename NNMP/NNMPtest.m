close all
warning('off','all')
dims = 50;
max_it = 20;
rng(0)

errUs = zeros(max_it,1);
errUsSlow = zeros(max_it,1);

errNNOMP = zeros(max_it,1);
errFNNOMP = zeros(max_it,1);
errFW = zeros(max_it,1);
errAMP = zeros(max_it,1);
errFC = zeros(max_it,1);
errLSQ = zeros(max_it,1);

size_cone = 100;

% M = randn(dims,dims);
% M = M'*M;
% 
%M = diag(rand(dims,1));
M= eye(dims);
it =50;

convNNOMP = zeros(it,max_it);
convFNNOMP = zeros(it,max_it);
convPWMP = zeros(it,max_it);
convSlowMP = zeros(it,max_it);
convFCMP = zeros(it,max_it);
convFW = zeros(it,max_it);
convAMP = zeros(it,max_it);
%convNNLSQ = zeros(it,max_it);
for outIt = 1:max_it
A = [zeros(dims,1) normc(rand(dims,size_cone))];
  aopt = rand(size_cone+1,1)*5;
numberOfFlips = 10;
idx = randi(dims,numberOfFlips,1);

y = A*aopt;
y(idx)=y(idx)*(-1);
rho = 10*norm(y);
alpha0FW = rand(size_cone+1,1);
x0FW=A*alpha0FW./sum(alpha0FW);
close all




[alpha,convPWMP(:,outIt)]  = NNMP(A,y,M,it,dims);
errUs(outIt) = ((y-A*alpha)'*M*(y-A*alpha));
[alpha,convFCMP(:,outIt)]  = FCNNMP(A,y,M,it,dims);
errFC(outIt) = ((y-A*alpha)'*M*(y-A*alpha));

[x_new, convSlowMP(:,outIt)] = NNMPslow(A,y,M,it,dims);
errUsSlow(outIt) = (y-x_new)'*M*(y-x_new);

[x_new,convNNLSQ(:,outIt),~] = lsqnonnegMy(A(:,2:end),y);
errLSQ(outIt)=(y-A(:,2:end)*x_new)'*M*(y-A(:,2:end)*x_new);

[x_new, convNNOMP(:,outIt)]= nnomp_canon(A(:,2:end),y,it);
errNNOMP(outIt)=(y-A(:,2:end)*x_new)'*M*(y-A(:,2:end)*x_new);

[x_new, convFNNOMP(:,outIt)]= fnnomp(A(:,2:end),y,it,0.00001);
errFNNOMP(outIt)=(y-A(:,2:end)*x_new)'*M*(y-A(:,2:end)*x_new);

[x_new, convFW(:,outIt)] = FW(A,y,M,it,dims,rho,x0FW);
errFW(outIt) = (y-x_new)'*M*(y-x_new);

[alpha, convAMP(:,outIt)] = AMP(A,y,M,it,dims);
errAMP(outIt) = ((y-A*alpha)'*M*(y-A*alpha));
end

[fxstar,imin] = min([mean(errUs),mean(errFC),mean(errUsSlow),mean(errLSQ),mean(errNNOMP),mean(errFNNOMP),mean(errFW),mean(errAMP)]);
figure
% convNnomp = mean(convNNOMP,2)-fxstar;
% convNnomp(convNnomp<0.001) = 0;
% semilogy(1:it,convNnomp,'LineWidth',4)
%semilogy(1:it,mean(convNNLSQ,2)-fxstar,'LineWidth',4)
convPw = mean(convPWMP,2)-fxstar;
convPw(convPw<0.001) =0;
semilogy(1:it,convPw,'LineWidth',4)
hold on

semilogy(1:it,mean(convSlowMP,2)-fxstar,'LineWidth',4)
convFC = mean(convFCMP,2)-fxstar;
convFC(convFC<0.001) = 0;
semilogy(1:it,convFC,'LineWidth',4,'Color',[0.1953, 0.8008, 0.1953])
semilogy(1:it,mean(convFW,2)-fxstar,'--','LineWidth',4)
semilogy(1:it,mean(convAMP,2)-fxstar,':','LineWidth',4)
semilogy(1:it,mean(convFNNOMP,2)-fxstar,'+','LineWidth',4)

xlabel('Iteration')
ylabel('Suboptimality')
legend('PWMP (Alg. 3)','NNMP (Alg. 2)','FCMP (Alg. 4)','FW','AMP (Alg. 3)','FNNOMP')
title('Synthetic data')
set(gca,'FontSize',24)
