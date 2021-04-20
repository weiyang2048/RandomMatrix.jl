export  normview

function normview(M::AbstractMatrix)
    M = norm.(M)
    heatmap(M, 
            yflip=true,
                c=cgrad(:binary, 50, categorical = true),
                ratio = 1)
end

