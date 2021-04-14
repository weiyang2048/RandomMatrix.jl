module RandomMatrix

using LinearAlgebra, StatsBase, Distributions

export RandomUnitaryMatrix, RandomOrthogonalMatrix , Haar

"""
```julia
RandomUnitaryMatrix(3)
3×3 Matrix{ComplexF64}:
 -0.173896+0.299143im  -0.173916+0.627558im  -0.666748-0.107907im
  0.338596+0.832271im  0.0660322-0.146868im    0.18828-0.362351im
 -0.202431-0.17881im    0.724265-0.159481im  -0.277831-0.547599im
```
"""
function RandomUnitaryMatrix(n::Int)
    A = randn(ComplexF64,n,n)
    Q = convert(Matrix,qr(A).Q)
    return Q
end
function RandomOrthogonalMatrix(n::Int)
    A = randn(n,n)
    Q = convert(Matrix,qr(A).Q)
    return Q
end

struct Haar <: ContinuousMatrixDistribution
    beta::Int
    n::Int
    Haar(beta,n) = beta in (1,2) ? new(beta,n) : error("Only take beta = 1 for orthogonal, beta = 2 for unitary")
end
Base.size(d::Haar) = d.n
Base.rand(d::Haar) = d.beta == 1 ? RandomOrthogonalMatrix(d.n) : RandomUnitaryMatrix(d.n)

export RandomNormSamplingMatrix


"""
```julia
RandomNormSamplingMatrix(A::Matrix, B=Matrix(A')::Matrix; k=0::Int)
```
Generate a random norm-squared sampling random Matrix `S`, E(SS')=I.
ASS'B is typically used to approximate AB.


# Examples
```julia
S = RandomNormSamplingMatrix(rand(2,3),rand(3,2),k=2)

3×2 Matrix{Float64}:
 0.0      0.0
 0.0      1.2439
 1.15342  0.0
```
"""
function RandomNormSamplingMatrix(A::Matrix,B=Matrix(A')::Matrix; k=0::Int)
    if k <= 0 
        k = floor(Int,size(A)[2]/2)
        print("k was <= $(0) or not provided, it is now set to $(k) by default")
    end
    
    n, m = size(A)
    if size(B)[1] != m
        return "The matrices can not be multiplied "
    else
        Lens = [norm(A[:,i])*norm(B[i,:]) for i = 1:m ]
        Total = sum(Lens)
        S = zeros(m,k)
        for (i,j) in enumerate([sample([1:m...], Weights(Lens) ) for i = 1 : k])
            S[j,i] = 1/sqrt(Lens[j]/Total*k) # normalizing so that E[SS']=I
        end
        return S
    end
end


end # module
