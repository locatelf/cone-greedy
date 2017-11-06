function [alpha,err] = AMP(A,y,M,it,dims)

x = zeros(dims,1);
alpha = zeros(size(A,2),1);
err = zeros(it,1);
cnt = 0;
cntt = zeros(it,1);
for i=1:it
    r = -M*(y-x);
    inner=A'*r;
    [~,idxZ]=min(inner);
    z = A(:,idxZ);
    [~,idxV]=max(inner.*(alpha>0));
    v = A(:,idxV);
    d = z;
    if dot(r,d)<= dot(r,-v)
        idxV = 1;
    else
        d = -v;
        idxZ = 1;
        cntt(i) =1;
    end
    if d == zeros(dims,1)
       break 
    end
    gamma = -dot(r,d)/norm(d)^2;
    [gamma,alpha,cnt] = clip(gamma,alpha,idxZ,idxV,cnt);
    x = x + gamma*(d);
    err(i)=((y-x)'*M*(y-x)); 
end