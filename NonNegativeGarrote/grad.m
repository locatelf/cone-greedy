function grad=grad(y,x,b,c,bias,n,p,lambda)
grad = zeros(p,1);
for j=1:p
    for i = 1:n
        a =x(i,:)*(c.*b)+bias;
        grad(j) = grad(j)+(sigmoid(a)-y(i))*x(i,j)*b(j);
    end
    grad(j) = grad(j)+lambda;
end

end
