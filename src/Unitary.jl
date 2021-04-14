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