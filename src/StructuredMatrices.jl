export randToeplitz, randHankel

"""
```julia
randToeplitz(d, n;  norm, hermitian)  where D<:S
```
- 
- The limiting spectral distribution of random Toeplitz matrices is still an open problem
"""
function  randToeplitz(d::D, n::Int;  norm = false::Bool, hermitian=true::Bool)  where D<:S
    if hermitian
      t = rand(d,n)
      t = [t[n:-1:2]'..., t...]
    else
      t = rand(d,2n-1)
    end
    M = zeros(eltype(d),n,n)
    for i in 1:n,j in 1:n
        M[i,j]=t[i-j+n]
    end

    if norm  return M/sqrt(n) end
    M

end

function randToeplitz(n::Int; norm=false::Bool, hermitian=true::Bool)
  return randToeplitz(Normal(),n,norm=norm,hermitian=hermitian)
end


"""
```julia
randHankel((1,im,pi),5)

5×5 Matrix{Number}:
  1   1  im  1   1
  1  im   1  1   π
 im   1   1  π   π
  1   1   π  π   π
  1   π   π  π  im
```

"""
function randHankel(d::D, n::Int;  norm = false::Bool)  where D<:S
  h = rand(d,2n-1)
  M = zeros(eltype(d),n,n)
  for i in 1:n
    M[i,:] = h[i:i+n-1]
  end

  if norm  return M/sqrt(n) end
  M
end

function randHankel(n::Int;  norm = false::Bool)
  return randHankel(Normal(),n,norm=norm)
end