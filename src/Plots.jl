export  normview

"""
```julia
# Example

# The resolvent of a random Hermitian matrix is approximately diagonal, 
# see Section 3 in https://arxiv.org/pdf/1903.10060.pdf
 normview(resolvent(randHermitian(1000,norm=true))(0.5+0.1im))
```
"""
function normview(M::AbstractMatrix)
    M = norm.(M)
    heatmap(M, 
            yflip=true,
                c=cgrad(:binary, 50, categorical = true),
                ratio = 1, 
                xlims=(0.5,size(M)[1]+0.5),ylims=(0.5,size(M)[2]+0.5)
                )
end

