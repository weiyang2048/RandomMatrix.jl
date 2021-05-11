
"""
```julia
randToeplitz(d, n;  norm, hermitian)  
```
- `d` : entry distribution
- `n` : dimension
- `norm` : default `false`;  if `norm` set to `true`, then the matrix will be normlaized with ``n^{-1/2}``.  
- `hermitian`: default `true`; if `true` the matrix will be Hermitian

# Examples

Generate a ``4 \\times 4`` random Hermitian Toeplitz matrix with entries Standard Normal.
```julia
randToeplitz(4)

4×4 Matrix{Float64}:
  1.10207   -0.47292   -0.745498   1.06809
 -0.47292    1.10207   -0.47292   -0.745498
 -0.745498  -0.47292    1.10207   -0.47292
  1.06809   -0.745498  -0.47292    1.10207
```
Generate a ``4 \\times 4`` normalized random Toeplitz matrix with entries `Exponential(1)`.
```julia
using Distributions
randToeplitz(Exponential(1),4, norm = true, hermitian = false)

4×4 Matrix{Float64}:
 0.667888  0.260045  1.48812   0.477305
 1.50374   0.667888  0.260045  1.48812
 1.1475    1.50374   0.667888  0.260045
 0.363966  1.1475    1.50374   0.667888
```
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
randHankel(d, n;  norm )  

randHankel(n;  norm)
```
- `d` : entry distribution
- `n` : dimension
- `norm` : default `false`,  if `norm` set to `true`, then the matrix will be normlaized with ``n^{-1/2}``.  

# Examples

Generate a ``5\\times 5`` random Hankel matrix with entries uniformly distributed on ``\\{1, i, \\pi \\}``
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