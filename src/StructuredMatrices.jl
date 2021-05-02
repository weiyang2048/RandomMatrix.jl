export randToeplitz

"""
```julia
randToeplitz(d::D, n::Int;  norm = false::Bool, hermitian=true::Bool, complex =false::Bool)  where D<:Any
```
- The limiting spectral distribution of random Toeplitz matrices is still an open problem
"""
function  randToeplitz(d::D, n::Int;  norm = false::Bool, hermitian=true::Bool, complex =false::Bool)  where D<:Any
    if hermitian
      t = rand(d,n)
      t = [t[n:-1:2]'..., t...]
    else
      t = rand(d,2n-1)
    end
    if complex
        M = zeros(ComplexF64,n,n)
    else
        M =zeros(n,n)
    end
    for i in 1:n,j in 1:n
        M[i,j]=t[i-j+n]
    end

    if norm  return M/sqrt(n) end
    M

end