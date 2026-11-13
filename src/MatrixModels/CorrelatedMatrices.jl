export randElliptic, randEllipic

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
            N = [rand(MvNormal([1 ρhat;ρhat 1])) for _=1:10000]
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
randElliptic(d, n; r , diag, norm) 

randElliptic(n; r, norm)
```
- `d` : default `Normal()`, entry distribution
- `n`  : dimensions 
- `r` : default `0.5`, the correlation of ``H_{ij},H_{ji}`` pairs; must lie in ``[-1, 1]``
  (`r = 1` gives a Hermitian matrix; `r = -1` is the exact antisymmetric case below)
- `norm` : default `false`, if `norm` set to `true`, then the matrix will be normalized with ``n^{-1/2}``.  
- `diag` : default `diag=d`, the distribution for diagonal entries.

The endpoint `r = -1` (perfect anticorrelation) cannot come from the
NORTA / bivariate-normal construction, which is singular there, so it is
built exactly instead: the upper triangle is drawn from `d`, the lower
triangle is its negation, and the diagonal is zero — the matrix is
exactly antisymmetric, ``A + A^{\\top} = 0``. That requires `d` to be
symmetric about 0 (a negated draw must have the same law as a draw:
centred `Normal`, `Rademacher`, `Uniform(-a, a)`, …); any other entry
law throws `ArgumentError`, since the anticorrelation is not realizable
with it. `diag` plays no role at `r = -1`: antisymmetry forces a zero
diagonal. The eigenvalues are purely imaginary, and with `norm = true`
their imaginary parts follow the semicircle law on ``[-2\\sigma, 2\\sigma]``,
where ``\\sigma^2`` is the variance of `d`.

# Examples
Generate a random elliptic matrix, with entries from ``\\mathscr{N}(0,1)`` and ``\\rho(H_{ij},H_{ji})=0.5`` 
```julia
randElliptic(500)

500×500 Matrix{Float64}:
  2.03417    -0.424289    1.28267   …  -0.114754  -1.96059
  0.44479    -1.45563     1.32828       1.00149    0.45786
  1.56525     0.0832211  -0.186738     -1.3914     1.04151
 -0.11633    -0.483301   -1.81348      -1.57536    0.514818
  ⋮                                 ⋱
  1.24274    -0.411623   -1.04984      -0.812778  -1.84479
 -0.0817287  -0.254886    0.674914      0.756269  -0.0296209
 -1.48281     0.51675    -1.58041       0.156923   0.244599
  0.852339    1.04593    -0.119082      1.43634    0.114493
```

Generate a normalized random elliptic matrix, with entries `Poisson(10)` and ``\\rho(H_{ij},H_{ji})=0.1`` 
```julia
using Distributions
randElliptic(Poisson(10),500, r=0.1 , norm=true)

500×500 Matrix{Float64}:
  0.268328    -0.0413153  -0.0175096   …   0.0190835   0.0201304       
 -0.00599949   0.447214   -0.0175878       0.0112805   0.100704        
 -0.0258879    0.0219927   0.402492       -0.04749    -0.050853        
  0.0219071    0.0119609  -0.00448502      0.0043233   0.0404757       
  ⋮                                    ⋱
 -0.0145467    0.0800297   0.00247891      0.0189267   0.071565        
  0.016412     0.0334019   0.0663348      -0.0180889   0.023773        
 -0.0485914   -0.0575288  -0.0409827       0.491935   -0.0969691       
  0.0405447    0.0503843   0.00624668      0.0558304   0.402492        

```
"""
function randElliptic(d::D, n::Int; r = 0.5::Float64, diag=d::D, norm = false::Bool) where  D<:S
    # r is a correlation, and the NORTA construction below needs the
    # bivariate normal with correlation rhohat to be positive definite:
    # validate at the boundary instead of leaking a PosDefException from
    # inside preNORTA / MvNormal (which is what |r| > 1 and non-finite r
    # produced before). The two endpoints are exact comparisons, the
    # same idiom as the r == 1 Hermitian branch: r = 1 delegates, and
    # r = -1 takes the exact antisymmetric construction below, because
    # the bivariate normal is singular at perfect anticorrelation. A
    # tolerance band around -1 would be wrong here: r slightly above -1
    # is a valid (if near-degenerate) NORTA input and must keep that
    # ensemble, not be silently rerouted to the exact one.
    (isfinite(r) && -1 <= r <= 1) ||
        throw(ArgumentError("randElliptic requires the pair correlation r in [-1, 1], got r = $r"))
    if r==1
        return randHermitian(d,n,diag=diag,norm=norm)
    end
    if r==-1
        return _randEllipticAntisymmetric(d,n,norm=norm)
    end

    r = preNORTA(d,r).ρhat
    M = zeros(n,n)
    for i in 1:n
        M[i,i] = rand(diag)
        for j in i+1:n
            M[i,j],M[j,i] = rand(MvNormal([1 r;r 1]))
        end
    end

    if norm
        M /= sqrt(n)
    end
    return M
end

# A law is symmetric about 0 when a negated draw has the same law as a
# draw. Checked on the support endpoints and a deterministic quantile
# grid (p = 0.5 excluded: for a discrete symmetric law such as
# Rademacher the 0.5-quantile sits on an atom and is not itself 0).
# Exact for the standard symmetric laws — centred Normal / Laplace /
# Cauchy / Student-t, Rademacher, Uniform(-a, a) — and it rejects
# shifted or one-sided laws (Poisson, Exponential, Normal(mu != 0), ...).
function _issymmetric_about_zero(d::Distribution)
    lo, hi = minimum(d), maximum(d)
    lo == -hi || return false
    for p in (0.001, 0.01, 0.05, 0.25, 0.4)
        qlo, qhi = quantile(d, p), quantile(d, 1 - p)
        abs(qlo + qhi) <= 1e-8 * max(1.0, abs(qlo), abs(qhi)) || return false
    end
    return true
end
_issymmetric_about_zero(d) = false

# The exact r = -1 case of randElliptic: upper triangle drawn from d,
# lower triangle its negation, diagonal zero. The pair (A_ij, A_ji) =
# (X, -X) has correlation exactly -1 and both marginals equal d, which
# is realizable only when d is symmetric about 0.
function _randEllipticAntisymmetric(d, n::Int; norm = false::Bool)
    _issymmetric_about_zero(d) ||
        throw(ArgumentError("randElliptic with r = -1 requires an entry distribution " *
            "symmetric about 0 (a negated draw must have the same law as a draw); got $d"))
    M = zeros(n,n)
    for i in 1:n
        for j in i+1:n
            M[i,j] = rand(d)
            M[j,i] = -M[i,j]
        end
    end

    if norm
        M /= sqrt(n)
    end
    return M
end

"""
(This is a old typo)
See [`randElliptic`](#randElliptic) for more details.
"""
function randElliptic(n::Int; r=0.5::Float64, norm = false::Bool)
    return randElliptic(Normal(),n, r=r, norm=norm)
end

const  randEllipic = randElliptic

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
randStochastic(3)

3×3 Matrix{Float64}:
 0.132593  0.216041  0.651367
 0.484097  0.320777  0.195126
 0.261495  0.537825  0.20068
```

Generates a 3 by 3 normalized random row stochastic  matrix
```julia
randStochastic(3,type = 1)

3×3 Matrix{Float64}:
 0.220849   0.146942  0.632209
 0.188052   0.26294   0.549008
 0.0170714  0.524574  0.458355
```

Generates a 3 by 3 normalized random column stochastic  matrix
```julia
randStochastic(3,type=2,norm=true)

3×3 Matrix{Float64}:
 0.583396  0.608739  0.732921
 0.672821  0.078786  0.302657
 0.475834  1.04453   0.696473
```
"""
function randStochastic(n::Int; type = 3 ::Int, norm = false::Bool)
    if type == 3
       M = randUnitary(n)
       M = M.*conj(M)
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