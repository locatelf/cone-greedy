function [a,b, objective]=LMO(R,Y,Xr,nil,LMO_it)

objective = sum(R(:).^2);

T = LMO_it; % max number of oracle iterations
rez = zeros(1,T-1);

%% random initialization of atoms
a = rand(size(Y,1),1);
a = a/norm(a);
b = rand(size(Y,2),1);
b = b/norm(b);



%% oracle
% stopping conditions
t = 1;
while t<T
    a_old = a;
    b_old = b;
   
    t = t + 1;
    a = R*b;
    a(a<0) = 0;
    if(norm(a)~=0)
    a = a/norm(a);
    end
    
    b = R'*a;
    b(b<0)=0;    
    if norm(b)~=0 
    b = b/norm(b);
    end

   
    Z = a*b';
    rez(t-1) = dot(-R(:),Z(:));

end
Z= a*b';
if dot(-R(:),Z(:))>0
    a = a.*0;
    b = b.*0;
end
end


