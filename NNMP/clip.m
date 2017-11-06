function [gamma,alpha,cnt] = clip(gamma,alpha,idxZ,idxV,cnt)
    if idxV~=1
       if alpha(idxV)-gamma<=0
           gamma = alpha(idxV);
           cnt = cnt +1;
       end
       alpha(idxV) = alpha(idxV) - gamma;
    end
    if idxZ~=1
        alpha(idxZ) = alpha(idxZ) + gamma;
    end
end