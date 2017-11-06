function [X_r, residual_time,test_error] = PWNMP(Y,X_0,R,nil,LMO_it,X_tst)

%% variables init
convergence_check = zeros(1,R+1);

X_r=X_0;

S = zeros(size(X_0));
alpha = 0;
test_error = zeros(R,1);

residual_time = zeros(1,R);
cnt = 0;
A = zeros(size(Y,1),1);
B = zeros(size(Y,2),1);
for r = 1:R
    r
    %% call to the oracle
    grad = -(Y-X_r);
    [a,b,residual] = LMO(-grad,Y,X_r,nil,LMO_it);
    residual_time(r) = residual;
    Z = a*b';
    A = cat(2,A,a);
    B = cat(2,B,b);
    maxV=0;
    idxV=1;
    idxZ=-1;
    for it=1:numel(alpha)
        c = S(:,:,it);
        if dot(grad(:),c(:))>maxV
            idxV = it;
            maxV = dot(grad(:),c(:));
        end
        
        if dot(Z(:),c(:)) == norm(Z(:))^2
            idxZ = it;
        end
    end
    if idxZ == -1
        idxZ = numel(alpha) + 1;
        S = cat(3,S,Z);
        alpha = [alpha,0];
    end
    V = S(:,:,idxV);
    D = Z - V;
    if dot(-grad(:),D(:)) == 0
       break 
    end
    
    gamma = dot(-grad(:),D(:))/norm(D(:));
    
    idxDelete = [];
    if idxV~=1
       if alpha(idxV)-gamma<=0
           gamma = alpha(idxV);
           cnt = cnt +1;
           idxDelete = idxV;
       else
       alpha(idxV) = alpha(idxV) - gamma;
       A(:,idxV) = A(:,idxV).*sqrt(alpha(idxV));
       B(:,idxV) = B(:,idxV).*sqrt(alpha(idxV));
       end
    end
    alpha(idxZ) = alpha(idxZ) + gamma;
    A(:,idxZ) = A(:,idxZ).*sqrt(alpha(idxZ));
    B(:,idxZ) = B(:,idxZ).*sqrt(alpha(idxZ));
    
    alpha(idxDelete) = [];
    S(:,:,idxDelete) = [];
    A(:,idxDelete) = [];
    B(:,idxDelete) = [];

    [Aprov, Bprov, ~] = NMF_GCD(Y,size(A,2)-1,20,A(:,2:end),B(:,2:end)',1);
    X_r = Aprov*Bprov;
    A = [zeros(size(X_r,1),1), normc(Aprov)];
    B = [zeros(size(X_r,2),1), normc(Bprov')];
    for itt = 1:size(A,2)
        S(:,:,itt) = sparse(A(:,itt)*B(:,itt)');
        %Sfull(:,:,itt) = Zprov;
    end
    
    alpha = [0,norms(Aprov).*norms(Bprov')];
    
    
    convergence_check(r) = 2*(sum(X_r(:).^2)-1);
    
    test_error(r) =  0.5*norm(X_tst - X_r,'fro')^2;
    r = size(S,3);

end
end

function S = norms(S)
    S = sqrt(sum(S.^2,1));
end