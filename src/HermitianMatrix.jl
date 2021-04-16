export randHermitian

"""
# Examples
- `randHermitian(2)` Generates a 2 by 2 random Symmetric matrix with entries from the real Gaussian(0,1).
- `randHermitian(ComplexF64,2)` Generates a 2 by 2 random orthogonal matrix with entries from the standard complex Gaussian.
- `randHermitian(Poisson(2),2)` Generate a random 2 by 2 Hermitian Matrix with entries  `Poisson(2)` rvs.  
  *Need to import the `Distributions` package for `Poisson(2)`*
"""
function randHermitian(d::T, n::Int, normalize = false::Bool) where T<:Union{Distribution{Univariate},DataType}
    M = zeros(typeof(rand(d)),n,n)
    for i in 1:n, j in 1:i-1
        M[i,j] = rand(d)
        M[j,i] = M[i,j]'
    end
    for i in 1:n
        M[i,i] = rand(d)
    end

    if normalize
        M/=sqrt(n)
    end

    return M
end

function randHermitian(n::Int, normalize = false::Bool)
    return randHermitian(Normal(), n, normalize)
end
