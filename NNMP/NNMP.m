function [alpha,err] = NNMP(A,y,M,it,dims)

x = zeros(dims,1);
alpha = zeros(size(A,2),1);
err = zeros(it,1);
cnt = 0;
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
