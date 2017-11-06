function [x, err] = NNMPslow(A,y,M,it,dims)

x = zeros(dims,1);
alpha = zeros(size(A,2),1);
err = zeros(it,1);
normalization = max(sqrt(sum(A(:,2:end).^2,1)));

for i=1:it
    r = -M*(y-x);
    Anew = [A,-x/norm(x)*normalization];
    inner=Anew'*r;
    [inner_min,idx]=min(inner);

    if(inner_min==0) break; end
    z = Anew(:,idx);

    gamma = -inner_min/(z'*M*z);

    x = x + gamma*z;
 
    err(i)=((y-x)'*M*(y-x));
        
end
