function [M1,M2,M3] = toMat (X)
    M1 = reshape(X,size(X,1),size(X,3)*size(X,2));
    M2 = reshape(permute(X,[2,1,3]), size(X,2), size(X,3)*size(X,1));
    M3 = reshape(permute(X,[3,1,2]), size(X,3), size(X,2)*size(X,1));
end