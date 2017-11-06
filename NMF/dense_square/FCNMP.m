function [X_r, residual_time,test_error] = FCNMP(Y,X_0,R,nil,LMO_it,X_tst)

%% variables init
convergence_check = zeros(1,R+1);

X_r=X_0;

S = sparse(zeros(numel(X_0),1));

test_error = zeros(R,1);

residual_time = zeros(1,R);
cnt = 0;
A = zeros(size(X_0,1),1);
B = zeros(size(X_0,2),1);
currentRank = 0;

while currentRank<R
    currentRank
    %% call to the oracle
    grad = -(Y-X_r);
    [a,b,residual] = LMO(-grad,Y,X_r,nil,LMO_it);
    S = cat(2,S,sparse(reshape(a*b',[],1)));
    A = cat(2,A,a);
    B = cat(2,B,b);

    a_new = lsqnonneg(S(:,2:end),Y(:));
    X_r = sparse(zeros(size(X_r)));
    idxRemove = [];
    for ii=2:numel(a_new)+1
        if a_new(ii-1)==0
            idxRemove = [idxRemove,ii];
        else
            X_r = X_r + sparse(a_new(ii-1)*A(:,ii)*B(:,ii)');
            A(:,ii) = A(:,ii).*sqrt(a_new(ii-1));
            B(:,ii) = B(:,ii).*sqrt(a_new(ii-1));
        end
    end
    
    S(:,idxRemove) = [];
    A(:,idxRemove) = [];
    B(:,idxRemove) = [];
    
    [Aprov, Bprov, ~] = NMF_GCD(Y,size(A,2)-1,20,A(:,2:end),B(:,2:end)',1);
    X_r = Aprov*Bprov;
    A = [zeros(size(X_r,1),1), normc(Aprov)];
    B = [zeros(size(X_r,2),1), normc(Bprov')];
    for itt = 1:size(A,2)
        S(:,itt) = sparse(reshape(A(:,itt)*B(:,itt)',[],1));
    end
    

    currentRank = size(A,2)-1;

    test_error(currentRank) =  0.5*norm(X_tst - X_r,'fro')^2;
save('workspace')
end

end

