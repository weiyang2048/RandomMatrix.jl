# RandomMatrix.jl

A package for Random Matrix Theory.

Hoping to add-in enough functionalities and publish in Aug 2021.


# Examples

## Random Matrix Theory

### Random Matrix Models
#### Random Unitary
```julia
RandomUnitaryMatrix(3)
3×3 Matrix{ComplexF64}:
 -0.173896+0.299143im  -0.173916+0.627558im  -0.666748-0.107907im
  0.338596+0.832271im  0.0660322-0.146868im    0.18828-0.362351im
 -0.202431-0.17881im    0.724265-0.159481im  -0.277831-0.547599im
```
This is equivalent to 

```julia
rand(Haar(2,3))
```
#### Random Orthogonal
```julia
RandomOrthogonalMatrix(3) 
```
or 
```julia
rand(Haar(1,3))
```

## Randomized Linear Algebra
```julia
RandomNormSamplingMatrix(A::Matrix, B=Matrix(A')::Matrix; k=0::Int)
```
Generate a random norm-squared sampling random Matrix `S`, ``\operatorname{E}(SS')=I``.
ASS'B is typically used to approximate AB.

```julia
S = RandomNormSamplingMatrix(rand(2,3),rand(3,2),k=2)

3×2 Matrix{Float64}:
 0.0      0.0
 0.0      1.2439
 1.15342  0.0
```