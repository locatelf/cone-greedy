function rez = to_vector_space(order_Y,n,atoms)
rez = 1;
    for i=order_Y:-1:1
        if i ~= n
            rez=kr(rez,atoms{i});
        end
    end
end