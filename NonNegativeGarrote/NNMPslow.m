function c=NNMPslow(y,x,b,bias,n,p,lambda,L)
it = 100;
c = zeros(p,1);
err = zeros(1,it);
for i=1:it
    r = grad(y,x,b,c,bias,n,p,lambda);
    
    atom = -r;
    atom(r>0)=0;
    atom = atom/norm(atom);
    Anew = [atom,-c/norm(c)];
    inner=Anew'*r;
    [inner_min,idx]=min(inner);
    
    if(inner_min==0) break; end
    z = Anew(:,idx);

    gamma = -inner_min/(L*norm(z)^2);

    c = c + gamma*z;
    for ix = 1:n
        a =x(ix,:)*(c.*b)+bias;
        err(i) = err(i)-y(ix)*log(sigmoid(a))-(1-y(ix))*log(1-sigmoid(a));
    end
    err(i) = err(i) + lambda*sum(c);
end

