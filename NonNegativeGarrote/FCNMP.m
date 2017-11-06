function c = FCNMP(y,x,b,bias,n,p,lambda,L)

%% variables init
R = 100;
c = zeros(p,1);

S = zeros(size(c));
alpha = 0;

err = zeros(1,R);
cnt = 0;
for it = 1:R
    %r
    %% call to the oracle
    r = grad(y,x,b,c,bias,n,p,lambda);
    
    z = -r;
    z(r>0)=0;
    z = z/norm(z);

    
    S = cat(2,S,z);
    if sum(abs(z)) == 0|| norm(r)<0.00000001
       break 
    end
    target = c-1/L*r;
    a_new = lsqnonneg(S(:,2:end),target);
    c = zeros(p,1);
    idxRemove = [];
    for ii=1:numel(a_new)
        if a_new(ii)==0
            idxRemove = [idxRemove,ii];
        end
        c = c + a_new(ii)*S(:,ii+1);
    end
    S(:,(idxRemove+1)) = [];
    a_new(idxRemove) = [];
    
    for ix = 1:n
        a =x(ix,:)*(c.*b)+bias;
        err(it) = err(it)-y(ix)*log(sigmoid(a))-(1-y(ix))*log(1-sigmoid(a));
    end
    err(it) = err(it) + lambda*sum(c);
end



end
