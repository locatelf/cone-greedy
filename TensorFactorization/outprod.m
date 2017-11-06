function rez = outprod(u,v,z)
    rez = zeros(size(u,1),size(v,1),size(z,1));
    partial = u*v';
    for k=1:size(z,1)
        rez(:,:,k) = partial*z(k);
    end
end