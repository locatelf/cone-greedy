function [alpha,err] = FCNNMP(A,y,M,it,dims)

x = zeros(dims,1);
alpha = zeros(size(A,2),1);
err = zeros(it,1);
cnt = 0;
S = x;
idxS = 1;
for i=1:it
    r = -M*(y-x);
    inner=A'*r;
    [~,idxZ]=min(inner);
    if idxZ == 1 || norm(r)<0.00000001 || ismember(idxZ,idxS)
        break
    end
    z = A(:,idxZ);
    S = [S,z];
    idxS = [idxS, idxZ];
    a_new = lsqnonneg(S(:,2:end),y);
    for innerIt = 1:numel(a_new)
        alpha(idxS(innerIt+1)) = a_new(innerIt);
    end
    x = A*alpha;
    err(i)=((y-x)'*M*(y-x)); 
end
err(i:end)= 10000000000000000;
    er_pre = 1000000000;
   for i = 1:it
       err(i) = min(err(i),er_pre);
       er_pre = err(i);
   end
end
