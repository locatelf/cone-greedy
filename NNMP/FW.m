function [x, err] = FW(A,y,M,it,dims,rho,x)

err = zeros(it,1);
A = A.*rho;
for i=1:it
    r = -M*(y-x);
    inner=A'*r;
    [~,idx]=min(inner);

    z = A(:,idx);

    gamma = min(1,-dot(r,z-x)/norm(z-x)^2);

    x = x + gamma*(z-x);
 
    err(i)=((y-x)'*M*(y-x));
        
end
