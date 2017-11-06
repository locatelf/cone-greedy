function rez = cast_to_set(rez,normalized,non_negative,sparsity)
    if strcmp(non_negative,'true')
        rez(rez<0)=0;
    end
    if ~strcmp(sparsity,'0')
        rez  = to_sparse(rez,str2double(sparsity));
    end 
    if strcmp(normalized,'true')
        rez=rez./norm(rez);
    end
end

function in = to_sparse(in,k)
    [~,sortIndex] = sort(abs(in(:)),'ascend');  %# Sort the values in ascending order  
    assert(size(sortIndex,1)-k>=0,'non zero falues greater than number of entries')
    minIndex = sortIndex(1:size(sortIndex,1)-k);  %# Get a linear index into A of the 5 largest values
    in(minIndex) = 0;
end