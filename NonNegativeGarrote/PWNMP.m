function c = PWNMP(y,x,b,bias,n,p,lambda,L)

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

    maxV=0;
    idxV=1;
    idxZ=-1;
    for itt=1:numel(alpha)
        vv = S(:,itt);
        if dot(r,vv)>maxV
            idxV = itt;
            maxV = dot(r,vv);
        end
        
        if dot(z,vv) == norm(z)^2
            idxZ = itt;
        end
    end
    if idxZ == -1
        idxZ = numel(alpha) + 1;
        S = cat(2,S,z);
        alpha = [alpha,0];
    end
    v = S(:,idxV);
    d = z - v;
    if d == zeros(size(z))
       break 
    end
    
    gamma = dot(-r,d)/(L*norm(d));
    
    alpha(idxZ) = alpha(idxZ) + gamma;

    if idxV~=1
       if alpha(idxV)-gamma<=0
           gamma = alpha(idxV);
           cnt = cnt +1;
           alpha(idxV) = [];
           S(:,idxV) = [];
       else
       alpha(idxV) = alpha(idxV) - gamma;
       end
    end
    
    
    c=c + gamma*d;
   
    for ix = 1:n
        a =x(ix,:)*(c.*b)+bias;
        err(it) = err(it)-y(ix)*log(sigmoid(a))-(1-y(ix))*log(1-sigmoid(a));
    end
    err(it) = err(it) + lambda*sum(c);
end



end

