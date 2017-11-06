function [X_r, residual_time,test_error] = GMP(Y,X_0,R,nil,LMO_it,X_tst)

    %% variables init
    convergence_check = zeros(1,R+1);
    
    a = rand(size(Y,1),1);
    a = a/norm(a);
    b = rand(size(Y,2),1);
    b=b/norm(b);
    X_0 = a*b';
    X_r=X_0;
    
    test_error = zeros(R,1);
    
    residual_time = zeros(1,R);
    for r = 1:R
        r
        %% call to the oracle
        grad = -(Y+1e-10)./(X_r+1e-10) +1;        
        
        [a,b,residual] = LMO(-grad,Y,X_r,nil,LMO_it);
        %residual_time(r) = residual;
        Z = a*b';
        if r~= 1
            origin = -X_r/norm(X_r(:));
            if dot(grad(:),origin(:))<dot(grad(:),Z(:))
                Z = origin;
            end
        end
        alpha = dot(-grad(:),Z(:))/100000;
        if alpha == 0
            
            break
        end

	X_r=X_r + alpha*Z;



    
       
        test_error(r) =  sum(sum(Y.*log(Y+1e-10)-Y.*log(X_r+1e-10)-Y+X_r));
        
    end
   
end

