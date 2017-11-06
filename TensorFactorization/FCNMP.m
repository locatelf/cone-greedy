function [X_r, residual_time,test_error] = FCNMP(Y,X_0,R,nil,LMO_it,idx_tst,X_tst)

%% variables init
convergence_check = zeros(1,R+1);

X_r=X_0;

S = zeros(numel(X_0),1);
Sfull = zeros(size(X_0));

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
    S = cat(2,S,Z(:));
    A = cat(2,A,a);
    B = cat(2,B,b);
    C = cat(2,C,c);

    Sfull = cat(4,Sfull,Z);
    if sum(abs(Z(:))) == 0|| norm(grad(:))<0.00000001
       break 
    end
    a_new = lsqnonneg(S(:,2:end),Y(:));
    X_r = zeros(size(X_r));
    idxRemove = [];
    for ii=2:numel(a_new)+1
        if a_new(ii-1)==0
            idxRemove = [idxRemove,ii];
            
        else
            X_r = X_r + a_new(ii-1)*cpdgen({A(:,ii),B(:,ii),C(:,ii)});
            A(:,ii) = A(:,ii).*nthroot(a_new(ii-1),3);
            B(:,ii) = B(:,ii).*nthroot(a_new(ii-1),3);
            C(:,ii) = C(:,ii).*nthroot(a_new(ii-1),3);

        end
        
    end
    S(:,idxRemove) = [];
    A(:,idxRemove) = [];
    B(:,idxRemove) = [];
    C(:,idxRemove) = [];
    
    initTensor = {};
    initTensor{1} = A(:,2:end);
    initTensor{2} = B(:,2:end);
    initTensor{3} = C(:,2:end);

    [X_pred,~] = ncp(tensor(Y),size(A,2)-1,'init',initTensor,'max_iter',40);
    X_r = double(tensor(X_pred));
    A = [zeros(size(X_r,1),1), normc(X_pred.U{1})];
    B = [zeros(size(X_r,2),1), normc(X_pred.U{2})];
    C = [zeros(size(X_r,3),1), normc(X_pred.U{3})];

    for itt = 1:size(A,2)
       S(:,itt) = kr(kr(C(:,itt),B(:,itt)),A(:,itt)); 
    end
    
    convergence_check(r) = 2*(sum(X_r(:).^2)-1);
    test_error(r) =  sqrt(mean((X_tst(:) - X_r(:)).^2));
    
end

end

