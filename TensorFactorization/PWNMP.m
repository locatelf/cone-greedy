function [X_r, residual_time,test_error] = PWNMP(Y,X_0,R,nil,LMO_it,idx_tst,X_tst)

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
C = zeros(size(Y,3),1);

for r = 1:R
    
    %% call to the oracle
    grad = -(Y-X_r);
    [a,b,c,residual] = LMO(Y,X_r,nil,LMO_it,idx_tst);
    residual_time(r) = residual;
    Z = cpdgen({a,b,c});
    A = cat(2,A,a);
    B = cat(2,B,b);
    C = cat(2,C,c);

    maxV=0;
    idxV=1;
    idxZ=-1;
    for it=1:numel(alpha)
        c = S(:,:,:,it);
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
        S = cat(4,S,Z);
        alpha = [alpha,0];
    end
    V = S(:,:,:,idxV);
    D = Z - V;
    if D == zeros(size(Z))
       break 
    end
    
    gamma = dot(-grad(:),D(:))/norm(D(:));
    
    idxRemove = [];
    if idxV~=1
       if alpha(idxV)-gamma<=0
           gamma = alpha(idxV);
           cnt = cnt +1;
           idxRemove = idxV;
       end
       alpha(idxV) = alpha(idxV) - gamma;
       A(:,idxV) = A(:,idxV).*nthroot(alpha(idxV),3);
       B(:,idxV) = B(:,idxV).*nthroot(alpha(idxV),3);
       C(:,idxV) = C(:,idxV).*nthroot(alpha(idxV),3);

       
    end
    alpha(idxZ) = alpha(idxZ) + gamma;
    A(:,idxZ) = A(:,idxZ).*nthroot(alpha(idxZ),3);
       B(:,idxZ) = B(:,idxZ).*nthroot(alpha(idxZ),3);
       C(:,idxZ) = C(:,idxZ).*nthroot(alpha(idxZ),3);

    
    
    alpha(idxRemove) = [];
           S(:,:,:,idxRemove) = [];
           A(:,idxRemove) = [];
           B(:,idxRemove) = [];
           C(:,idxRemove) = [];
    
     initTensor = {};
    initTensor{1} = A(:,2:end);
    initTensor{2} = B(:,2:end);
    initTensor{3} = C(:,2:end);
    [X_pred,~] = ncp(tensor(Y),size(A,2)-1,'init',initTensor,'max_iter', 40);
    X_r = double(tensor(X_pred));
    A = [zeros(size(X_r,1),1), normc(X_pred.U{1})];
    B = [zeros(size(X_r,2),1), normc(X_pred.U{2})];
    C = [zeros(size(X_r,3),1), normc(X_pred.U{3})];
    alpha = [0,X_pred.lambda'];
    
    
    for itt = 1:size(A,2)
       S(:,:,:,itt) = cpdgen({A(:,itt),B(:,itt),C(:,itt)}); 
    end
    
    convergence_check(r) = 2*(sum(X_r(:).^2)-1);
    test_error(r) =  sqrt(mean((X_tst(:) - X_r(:)).^2));
    r = size(A,2)-1;
end

end

