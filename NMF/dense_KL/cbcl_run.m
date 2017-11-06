clear
max_iter = [100 1200];
rand('seed',0);
load '../data/cbcl';
n=size(V,1);
m=size(V,2);
K = 10;
err0 = zeros(1,K);
err1 = zeros(1,K);
for k = 2:K
times = 1;
objlist0 = zeros(1,max_iter(1));
objlist1 = zeros(1,max_iter(2));
timelist0 = zeros(1,max_iter(1));
timelist1 = zeros(1,max_iter(2));
for i=1:times
	W = abs(rand(n,k));
	H = abs(rand(k,m));
	obj_ini = KL(V,W,H);
 	[w h obj0 time0] = KLnmf(V,k,max_iter(1),W',H, 1);
 	[w1 h1 obj1 time1] = multKL(V,k,max_iter(2), W, H );
%     
    
	objlist0 = objlist0 + obj0;
	objlist1 = objlist1 + obj1;
	timelist0 = timelist0 + time0;
	timelist1 = timelist1 + time1;
end
err0(k) = objlist0(end)/times;
err1(k) = objlist1(end)/times;
% test_error(end)

end
 X0 = zeros(size(V));
 [X_pred, residual,test_error] = FCNMP(V,X0,k,0,100,V);
  [X_pred, residual,test_errorPW] = PWNMP(V,X0,k,0,100,V);

 semilogy(2:k-1, err0(2:end-1)-test_error(end),'LineWidth',5)
hold on
   semilogy(2:k-1, test_error(2:end-1)-test_error(end),'LineWidth',5)
 semilogy(2:k-1, err1(2:end-1)-test_error(end),'--','LineWidth',5)
 semilogy(2:k-1, test_errorPW(2:end-1)-test_error(end),':','LineWidth',5)

legend('CCD','FCMP','multiplicative','PWMP')
xlabel('Rank')
ylabel('Suboptimality')
set(gca,'FontSize',24)
