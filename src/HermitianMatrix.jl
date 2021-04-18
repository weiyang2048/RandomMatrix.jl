export randHermitian

"""
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
function randHermitian(d::T, n::Int, normalize = false::Bool)  where T<:Union{Distribution{Univariate}
                                                                 ,DataType,AbstractArray, Tuple}
    M = zeros(Number,n,n)
    for i in 1:n, j in i:n
        M[i,j] = rand(d)
    end

    if normalize
        M/=sqrt(n)
    end

    return Hermitian(M)
end

function randHermitian(n::Int, normalize = false::Bool)
    return randHermitian(Normal(), n, normalize)
end
