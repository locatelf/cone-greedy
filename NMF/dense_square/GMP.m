function [X_r, residual_time,test_error] = GMP(Y,X_0,R,nil,LMO_it,idx_tst,X_tst)

    %% variables init
    convergence_check = zeros(1,R+1);
    
    
    X_r=X_0;
    
    test_error = zeros(R,1);
    
    residual_time = zeros(1,R);
    for r = 1:R
        %r
        %% call to the oracle
        [a,b,residual] = LMO(Y,X_r,nil,LMO_it,idx_tst);
        residual_time(r) = residual;
        Z = a*b';
        grad = -(Y-X_r);        
        if r~= 1
            origin = -X_r/norm(X_r(:));
            if dot(grad(:),origin(:))<dot(grad(:),Z(:))
                Z = origin;
            end
        end
        alpha = dot(-grad(:),Z(:));
        if alpha == 0
            
            break
        end

	X_r=X_r + alpha*Z;



    
        convergence_check(r) = 2*(sum(X_r(:).^2)-1);
        %test_error(r) =  sqrt(mean((X_tst(idx_tst) - X_r(idx_tst)).^2));
        test_error(r) =  sqrt(mean((X_tst(:) - X_r(:)).^2));
        
    end
   
end

