export randHermitian

"""
```julia
randHermitian(d::T, n::Int; norm = false::Bool) where T <: Union{Distribution{Univariate}, DataType, AbstractArray, Tuple}
randHermitian(n::Int; norm = false::Bool)
```
If `norm` set to `true`, then the matrix will be normlaized with n^(-1/2).  
# Examples
- ```julia
randHermitian(2)
``` 
Generates a 2 by 2 random Symmetric matrix with entries from the real Gaussian(0,1).
- ```julia
randHermitian(ComplexNormal(),2)
``` 
Generates a 2 by 2 random orthogonal matrix with entries from the standard complex Gaussian.
- ```julia
randHermitian(Poisson(2),2)
``` 
Generate a random 2 by 2 Hermitian Matrix with entries  `Poisson(2)` rvs.  
  *Need to import the `Distributions` package for `Poisson(2)`*
- ```julia
randHermitian(1:10,2)
``` 
- ```julia
randHermitian([-1,pi],2)
```
"""
function randHermitian(d::T, n::Int; norm = false::Bool)  where T<:Union{Distribution{Univariate}
                                            ,DataType,AbstractArray, Tuple}


    if occursin("Complex",string(typeof(rand(d)))) # to walk around the InexactError
        M = zeros(ComplexF64,n,n)                  # when using eigvals(M)
    else    
        M = zeros(Number,n,n)
    end

    for i in 1:n, j in i:n
        M[i,j] = rand(d)
    end

    if norm
        M/=sqrt(n)
    end

    return Hermitian(M)
end

function randHermitian(n::Int; norm = false::Bool)
    return randHermitian(Normal(), n; norm = norm)
end
