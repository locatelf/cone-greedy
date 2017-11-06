function [a,b,c, objective]=LMO(Y,Xr,nil,LMO_it,idx)

R = Y-Xr; %compute residual
objective = sum(R(:).^2);

R1 = tens2mat(R,1);
R2 = tens2mat(R,2);
R3 = tens2mat(R,3);

T = LMO_it; % max number of oracle iterations
rez = zeros(1,T-1);

%% random initialization of atoms
a = rand(size(Y,1),1);
a = a/norm(a);
b = rand(size(Y,2),1);
b = b/norm(b);
c = rand(size(Y,3),1);
c = c/norm(c);


%% oracle
% stopping conditions
t = 1;
while t<T
    a_old = a;
    b_old = b;
    c_old = c;
   
    t = t + 1;
    a = R1*prova(c,b);%R1*kron(c_old,b_old);
    a(a<0) = 0;
    if(norm(a)~=0)
    a = a/norm(a);
    end
    
    b = R2*prova(c,a);
    b(b<0)=0;    
    if norm(b)~=0 
    b = b/norm(b);
    end

    c = R3*prova(b,a);
    c(c<0) = 0;
    if norm(c)~=0
    c = c/norm(c);
    end
    Z = cpdgen({a,b,c});
    rez(t-1) = dot(-R(:),Z(:));
    if t~=2
       if rez(t-1)>rez(t-2)
           a = a_old;
           b=b_old;
           c=c_old;
           break
       end
    end
end
Z= cpdgen({a,b,c});
% close all
% plot(1:T-1,rez)
if dot(-R(:),Z(:))>0
    a = a.*0;
    b = b.*0;
    c = c.*0;
end



