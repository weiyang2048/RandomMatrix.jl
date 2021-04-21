export  normview

function normview(M::AbstractMatrix)
    M = norm.(M)
    heatmap(M, 
            yflip=true,
                c=cgrad(:binary, 50, categorical = true),
                ratio = 1, 
                xlims=(0.5,size(M)[1]+0.5),ylims=(0.5,size(M)[2]+0.5)
                )
end

