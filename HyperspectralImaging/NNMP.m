function [alpha,err,cnt,i] = NNMP(A,y,M,it,dims,cnt)

x = zeros(dims,1);
alpha = zeros(size(A,2),1);
err = zeros(it,1);

for i=1:it
    r = -M*(y-x);
    inner=A'*r;
    [~,idxZ]=min(inner);
    z = A(:,idxZ);
    [~,idxV]=max(inner.*(alpha>0));
    v = A(:,idxV);
    d = z-v;
    if d == zeros(dims,1)
       break 
    end
    gamma = -dot(r,d)/(d'*M*d);
    [gamma,alpha,cnt] = clip(gamma,alpha,idxZ,idxV,cnt);
    x = x + gamma*(z-v);
    err(i)=((y-x)'*M*(y-x)); 
end
%plot(1:it,err)
 
% xlabel('iteration','FontSize', 30)
% ylabel('$\varepsilon_t$','Interpreter','LaTex','FontSize', 30)