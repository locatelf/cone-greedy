function [X_r, residual_time,test_error] = FCNMP(Y,X_0,R,nil,LMO_it,X_tst)

%% variables init
convergence_check = zeros(1,R+1);
 a = rand(size(Y,1),1);
    a = a/norm(a);
    b = rand(size(Y,2),1);
    b=b/norm(b);
    Z = a*b';
    S = zeros(numel(Z),1);
Sfull = zeros(size(Z));
    A = zeros(size(Z,1),1);
    B = zeros(size(Z,2),1);

    S = cat(2,S,Z(:));
    Sfull = cat(3,Sfull,Z);
    A = cat(2,A,a);
    B = cat(2,B,b);
X_r=Z;


alpha = 0;
test_error = zeros(R,1);

residual_time = zeros(1,R);
cnt = 0;
%for r = 1:R
r = 1;
test_error(r) =  sum(sum(Y.*log(Y+1e-10)-Y.*log(X_r+1e-10)-Y+X_r));
while r<R
    r
    %% call to the oracle
    grad = -(Y+1e-10)./(X_r+1e-10) +1;%-(Y-X_r);
    [a,b,residual] = LMO(-grad,Y,X_r,nil,LMO_it);
    %residual_time(r) = residual;
    Z = a*b';
    S = cat(2,S,Z(:));
    Sfull = cat(3,Sfull,Z);
     A = cat(2,A,a);
    B = cat(2,B,b);
    if sum(abs(Z(:))) == 0|| norm(grad(:))<0.00001
       break 
    end
    a_new = lsqnonneg(S(:,2:end),X_r(:)-0.1*grad(:));
    X_r = zeros(size(X_r));
    idxRemove = [];
    for ii=2:numel(a_new)+1
        if a_new(ii-1)<=0
            idxRemove = [idxRemove,ii];
        else
            X_r = X_r + a_new(ii-1)*Sfull(:,:,ii);
            A(:,ii) = A(:,ii).*sqrt(a_new(ii-1));
            B(:,ii) = B(:,ii).*sqrt(a_new(ii-1));

        end
    end
    
    
    S(:,idxRemove) = [];
    Sfull(:,:,idxRemove)=[];
    A(:,idxRemove) = [];
    B(:,idxRemove) = [];
    
    if 1%r==R-1
    [Aprov,Bprov,~]=multKL(Y,size(A,2)-1,240,A(:,2:end),B(:,2:end)');
    X_r = Aprov*Bprov;
    A = [zeros(size(Z,1),1), normc(Aprov)];
    B = [zeros(size(Z,2),1), normc(Bprov')];
    for itt = 1:size(A,2)
        Zprov = A(:,itt)*B(:,itt)';
        S(:,itt) = Zprov(:);
        Sfull(:,:,itt) = Zprov;
    end
    end
    r = size(S,2)-1;
    
    

    
    test_error(r) =  sum(sum(Y.*log(Y+1e-10)-Y.*log(X_r+1e-10)-Y+X_r));

end

end

