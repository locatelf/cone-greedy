function [x, errors] = fnnomp(A,y,IT,corr_max)
% Non-Negative Orthogonal Matching Pursuit with a real value dictionary

r = y;
[m,n] = size(A);
x = zeros(n,1);
k = 1;
mag = 1;
s = [];
bpr = A'*r;
Q = zeros(m,IT);
R = zeros(IT);
Rm1 = [];
xs = [];
r_pre = zeros(size(r));
tol = corr_max;
part = zeros(n,1);
errors = zeros(IT,1);

while k <= IT && mag>0 && abs(norm(r_pre,2) - norm(r,2)) >= tol 
    done = 0;
    zc = 0;
    Inc = [];
    l = 1;   
        bpr(s) = 0;       
        [amp,In] = sort(bpr,'descend');
    while ~done
        if amp(l) > 0       
            sint = [s;In(l)];
            anew = A(:,sint(end));
        
            qP=Q(:,1:k-1)'*anew;
            q=anew-Q(:,1:k-1)*(qP);
            nq=norm(q);
            q=q/nq;
            zin = q'*y;
            
            %%% Positivity Guarantee
        
            v = qP;
            mu = nq;
            gamma = -Rm1*v/mu;
            
            xsp = xs;  
            if ~isempty(gamma)
            if ~isempty(find(gamma<0, 1))    
                vt = abs((xsp./gamma).*(gamma<0));                
                zt = min(vt(vt>0));
            else
                zt = inf;
            end
            else
                zt = inf;
            end            
            if (zin <= zt) 
                if (zin <= zc)                                                                   
                    s = [s;In(l)];
                    anew = A(:,In(l));       
                    qP=Q(:,1:k-1)'*anew;
                    q=anew-Q(:,1:k-1)*(qP);
                    nq=norm(q);
                    q=q/nq;
                else
                    s = [s;In(l)];
                end
                done = 1;
            else
                if zc>=zin
                    s = [s;In(lc)];
                    anew = A(:,In(lc));       
                    qP=Q(:,1:k-1)'*anew;
                    q=anew-Q(:,1:k-1)*(qP);
                    nq=norm(q);
                    q=q/nq;
                    done = 1;
                elseif zt > zc,                                 
                    zc = zt;
                    lc = l;
                    l = l+1;
                else
                    l = l+1;
                end                                     
            end                                                        
        else
            done = 1;
            mag = amp(l);
        end
        if done && (mag > 0)
%              R(1:k-1,k)=qP; % Updatin R
%              R(k,k)=nq; % Updatin R
             Q(:,k)=q;
             Rm1 = [Rm1,gamma;zeros(1,size(Rm1,2)),1/mu];
             z(k)=q'*y;
             r_pre = r;
             r = r- q*z(k);
             xs = Rm1*z(1:k)';
        end
    end    
    bpr = A'*r;   
    part(s) = xs;
    errors(k)=norm(y-A*part)^2;
    k = k+1;
end
errors(errors == 0)= 10000000000;
if k~=IT
    er_pre = 10000000000000;
   for i = 1:IT
       errors(i) = min(errors(i),er_pre);
       er_pre = errors(i);
   end
end
x = part;
