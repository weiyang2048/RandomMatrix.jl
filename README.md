# RandomMatrix.jl

A package for Random Matrix Theory.

Hoping to add-in enough functionalities and publish in Aug 2021.


# Examples

## Random Matrix Theory

### Random Matrix Models
#### Random Unitary / Orthogonal
- To generate a 3 by 3 random Unitary matrix run 
    - `RandomUnitaryMatrix(3)`  or equivalently
    - `rand(Haar(2,3))`
- To generate a 3 by 3 random Orthogonal matrix run 
    - `RandomOrthogonalMatrix(3)` or equivalently
    - `rand(Haar(1,3))`
```

## Randomized Linear Algebra
```julia
RandomNormSamplingMatrix(A::Matrix, B=Matrix(A')::Matrix; k=0::Int)
```
Generate a random norm-squared sampling random Matrix `S`, ``E(SS')=I``.
ASS'B is typically used to approximate AB.

```julia
S = RandomNormSamplingMatrix(rand(2,3),rand(3,2),k=2)

3×2 Matrix{Float64}:
 0.0      0.0
 0.0      1.2439
 1.15342  0.0
```