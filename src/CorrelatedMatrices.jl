export randEllipic,randStochastic

"""
-  Look for a `phat`, such that when used in NORTA, it will return correlations approximately `ρ`
- `d` : distribution
- `ρ` : target correlation
- NORTA method for generating Random Vectors with Arbitrary Marginal Distribution 
    fand CorrelationMatrices first appears in the following paper
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.48.281&rep=rep1&type=pdf
    The generated vector entries usually have a smaller correlation than what is aimed for.
"""
struct preNORTA 
    d::D where D<:S
    ρ::Float64
    ρhat::Float64
    function preNORTA(d,ρ,ρhat=ρ)
        L , H = minmax(sign(ρ),0) # Lower bound, Higher bound
        ϵ = min(0.01,(1-abs(ρ))/10)
        while true
            N = [rand(MvNormal([1 ρhat;ρhat 1])) for i=1:10000]
            X = []
            Y = []

            for i in N
               append!(X,quantile(d,cdf(Normal(),i[1]))) 
               append!(Y,quantile(d,cdf(Normal(),i[2]))) 
            end
             
            temp = cor(X,Y)
            if abs(temp-ρ) < ϵ || H - L < ϵ
                break
            end
            if temp < ρ
                ρhat, L = (ρhat+H)/2 ,  (L+ρhat*4)/5
            else
                ρhat, H = (ρhat+L)/2 , (H+ρhat*4)/5
            end
        end
        new(d,ρ,ρhat)
    end
end 
"""
```julia
randEllipic(d::D, n::Int; r = 0.5::Float64, Diag=d::T, norm = false::Bool) where  D<:S

randEllipic(n::Int; r=0.5::Float64, norm = false::Bool)
```
- `d` : default `Normal()`, entry distribution
- `n`  : dimensions 
- `r` : default `0.5`, the correlation of ``H_{ij},H_{ji}`` pairs
- `norm` : default `false`, if `norm` set to `true`, then the matrix will be normalized with n^(-1/2).  
- `Diag` : default `Diag=d`, the distribution for diagonal entries.
    To use a different distribution (say Binomial) for digonal elements, set `Diag = Binomial(1,0.5)`

```julia
# Examples

# Generate a normalized random elliptic matrix, with correlation 0.5
A = randEllipic(500,norm=true)

# Plot the eigenvalues of A, this should look like an ellipse
A|>eigvals|>scatter

# Generate a normalized random elliptic matrix, with entries `Poisson(10)` correlation 0.1
using Distributions
randEllipic(Poisson(10),500, r=0.1 , norm=true)
```
"""
function randEllipic(d::D, n::Int; r = 0.5::Float64, Diag=d::D, norm = false::Bool) where  D<:S
    if r==1
        return randHermitian(d,n,diag=diag,norm=norm)
    end
    # TODO: take care of the r=-1 case.

    r = preNORTA(d,r).ρhat
    M = zeros(n,n)
    for i in 1:n
        M[i,i] = rand(Diag)
        for j in i+1:n
            M[i,j],M[j,i] = rand(MvNormal([1 r;r 1]))
        end
    end

    if norm
        M /= sqrt(n)
    end
    return M
end

function randEllipic(n::Int; r=0.5::Float64, norm = false::Bool)
    return randEllipic(Normal(),n, r=r, norm=norm)
end

"""
```julia
randStochastic(n; type, norm)
```
- `n`: dimension
- `type` : default `type = 3`, `3` for doubly randStochastic, `1` for row, `2` for column stochastic 
- `norm` : default `false`, if set to `true`, the matrix will be normalized by ``\\sqrt{n}`` (*not a typo*)
# Examples 

Generates a 3 by 3 random doubly stochastic  matrix
```julia 
julia> randStochastic(3)

3×3 Matrix{Float64}:
 0.132593  0.216041  0.651367
 0.484097  0.320777  0.195126
 0.261495  0.537825  0.20068
```
Generates a 3 by 3 normalized random column stochastic  matrix
```julia
julia> randStochastic(3,type=2,norm=true)

3×3 Matrix{Float64}:
 0.583396  0.608739  0.732921
 0.672821  0.078786  0.302657
 0.475834  1.04453   0.696473
```
"""
function randStochastic(n::Int; type = 3 ::Int, norm = false::Bool)
    if type == 3
       M = randUnitary(500)
       M = M.*conj(M)*sqrt(n)
    else 
        M = rand(n,n)
        for i in 1:n
            M[i,:] /= sum(M[i,:])
        end
        if type == 2
            M = M'
        end
    end
    
    if norm 
        M*=sqrt(n)
    end

    return M
end