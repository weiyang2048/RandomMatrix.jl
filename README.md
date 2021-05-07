A package for Random Matrix Theory.  The documentation can be found at ***[Documemtation🔗](https://weiyang2048.github.io/RandomMatrix.jl/dev/)***.


The package currently is unregister, you can only add it by
```julia
using Pkg
Pkg.add("https://github.com/weiyang2048/RandomMatrix.jl")
```
I hope to add in enough functionalities and register it in Aug 2021.
If there is any functionality you want me to implement, please raise an issue.

***[Documemtation🔗](https://weiyang2048.github.io/RandomMatrix.jl/dev/)***
> Examples will be slowly moving to the documentation page (still creating)  05/03/21
> 
> The following list will not be receiving any update   05/03/21
> 
> names not listed below:  randDiagonal, randTriangular, GOE, GUE   (05/03/21)
- [Examples](#examples)
  - [Random Matrix Theory](#random-matrix-theory)
    - [Random Matrix Models](#random-matrix-models)
      - [IID](#iid)
    - [Hermitian](#hermitian)
      - [Elliptic Matrices](#elliptic-matrices)
      - [Structured](#structured)
        - [stochastic](#stochastic)
        - [Toeplitz Matrices](#toeplitz-matrices)
      - [Unitary Matrices](#unitary-matrices)
        - [Unitary](#unitary)
        - [Orthogonal](#orthogonal)
        - [Haar](#haar)
        - [Permutation](#permutation)
    - [Transformations](#transformations)
      - [Resolvent](#resolvent)
      - [Quaternion Resolvent](#quaternion-resolvent)
  - [Distributions](#distributions)
    - [Complex Gaussian](#complex-gaussian)
      - [Gaussian](#gaussian)
    - [Circular Law](#circular-law)
    - [Elliptical Law](#elliptical-law)
    - [Marchenko-Pastur Law](#marchenko-pastur-law)
  - [Randomized Linear Algebra](#randomized-linear-algebra)
    - [Sampling Matrix](#sampling-matrix)
  - [Utility functions](#utility-functions)
    - [Operations](#operations)
      - [plus-minus](#plus-minus)
    - [Graphics](#graphics)
      - [Norm View](#norm-view)

# Examples

## Random Matrix Theory

### Random Matrix Models

#### IID
see ***[Documemtation🔗](https://weiyang2048.github.io/RandomMatrix.jl/dev/)***
### Hermitian
see ***[Documemtation🔗](https://weiyang2048.github.io/RandomMatrix.jl/dev/)***
#### Elliptic Matrices

```julia
randEllipic(d::D, n::Int; r = 0.5::Float64, Diag=d::D, norm = false::Bool) where D<:any

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
#### Structured
##### stochastic
see ***[Documemtation🔗](https://weiyang2048.github.io/RandomMatrix.jl/dev/)***
##### Toeplitz Matrices
```julia
randToeplitz(d::D, n::Int;  norm = false::Bool, hermitian=true::Bool, complex =false::Bool)  where D<:Any
```
- The limiting spectral distribution of random Toeplitz matrices is still an open problem

#### Unitary Matrices 
##### Unitary
```julia
randUnitary(n::Int)
```
- Generates a n by n random Unitary matrix
- Equivalent to run `rand(Haar(2,n))`
- For orthogonal matrices, use `randOrthogonal` or `rand(Haar(1,n))` instead

```julia
# Examples

# Generate a 3 by 3 random Unitary matrix 
randUnitary(3)
# or
rand(Haar(2,3)) 
```
##### Orthogonal
```julia
randOrthogonal(n::Int)
```
- Generates a n by n random Orthogonal matrix
- Equivalent to run `rand(Haar(1,n))`
- For unitary matrices, use `randUnitary` or `rand(Haar(2,n))` instead
 
```julia
# Examples

# Generates a 3 by 3 random Orthogonal matrix 
randOrthogonal(3)
# or
rand(Haar(1,3))
```
##### Haar
```julia
Haar(beta,n)
```
- Uniform distribution on O(n) (`beta = 1`), U(n) (`beta = 2`)
- `beta`: 1 for Orthogonal, 2 for Unitary
- `n`: dimension

```julia
# Examples

# Generate a 100 by 100 random Unitary Matrix uniformly from U(n)
rand(Haar(2,100))


# Generate a 100 by 100 random Orthogonal Matrix uniformly from O(n)
rand(Haar(1,100))
```
##### Permutation
```julia
randPermutation(n; fix) 
```
- `n` : dimension
- `fix` : a keyword argument, default is set to `fix = 0`. If `fix = x`, `randPermutation(n,x)` will have atleast `x` fixed points.
 
```julia
# Examples 

# Generates a random 5 by 5 permutation matrix
randPermutation(5)

# Generates  Generates a random 100 by 100 permutation matrix with atleast 10 fix points
randPermutation(100,fix=10)
```

### Transformations
#### Resolvent
```julia
resolvent(A::Matrix)
```

- return the resolvent ***function*** of `A`, ``R(z)= (A-zI)^{-1}``

```julia
# Examples

# This returns the resolvent **function** of a 500 by 500 Hermitian 
resolvent(randHermitian(500, norm = true)) # this is a generic function

# One can use the above return as a function
f = resolvent(randHermitian(500, norm = true))
f(0+0.1im) # this returns a matrix
```
#### Quaternion Resolvent
```julia
qresolvent(A)
```
>returns the quaternion resolvent ***function*** of A.

## Distributions
### Complex Gaussian
```julia
ComplexNormal(μ,σ)
```
- `μ` : mean, `μ = 0` by default
- `σ` : standard deviation, `σ  =1 ` by default

```julia 
# Examples

# Generates 10 iid standard Complex Gaussian r.v.s
rand(ComplexNormal(), 10) 

# Generates complex normal with mean 1+1im, variance 4
rand(ComplexNormal(1+1im,2)) 
```
#### Gaussian
```julia
Gaussian(beta,μ,σ)
```
- `beta` : 1 (default) for Real Gaussian, 2 for Complex Gaussian 
- `μ` : mean, `μ = 0` by default
-  `σ` : standard deviation, `σ  =1 ` by default

```julia 
# Generates 10 iid standard normal r.v.s
rand(Gaussian(), 10) 

# Generates a complex normal with mean 1+1im, variance 4
rand(Gaussian(2,1+1im,2))
```
### Circular Law
```julia
Circular(c=0,R=1)
```
- The uniform distribution on the complex disk with center `c` and radius `R`
***

```julia
rand(Circular(1,10),200)
```
> Generates 200 uniform r.v.s on the complex disk of center 1 radius 10.

### Elliptical Law

```julia
Elliptic(ρ=0.5,c=0,R=1)
```
- the uniform rv on an ellipse of width `2(1+ρ)R`, height `2(1-ρ)R` centered at `c`

### Marchenko-Pastur Law
```julia
MarchenkoPastur(λ,σ)
```
- Marchenko-Pastur r.v. with  asymptotic ratio λ and scale parameter σ.
- λ by default is 0.5
- σ by default is 1, the value 1 for σ is typically used in Random Matrix Theory.

```julia
pdf(d::MarchenkoPastur,x::Real)  
```

```julia
# Examples 

# Generate a MP rv with ρ = 0.5, σ =1
rand(MarchenkoPastur()) 

# Generate a 100 by 100 matrix with entries i.i.d MP rvs with ρ=0.2
rand(MarchenkoPastur(0.2),100,100)

# Generate 100 MP rvs with ρ=0.1 and σ=2
rand(MarchenkoPastur(0.1,2),100)

# Compute the desity for the MP distribution with λ=1.6 at the point 0
pdf(MarchenkoPastur(1.6),0)
```

## Randomized Linear Algebra
### Sampling Matrix
 ```julia
 randSampling(A,B,k=2)
 ```  
>If A is a n by m matrix and B is a m by w matrix.  Run `randSampling(A,B,k=2)` will generate a random  sampling matrix S of size m by k. Where E(SS')=I and 
  that E(ASS'B)=AB.  For definition, check the code or look for definition S:=SD at the end of section 2.2 (right before section 2.3) in https://arxiv.org/pdf/1608.04481.pdf.
##  Utility functions
### Operations
#### plus-minus
```julia
±(a,b)
```
- returns (a-b,a+b)
```julia
# Examples

1 ± 0.5  # returns (0.5,1.5)
```
### Graphics
#### Norm View
```julia
normview(M::AbstractMatrix)
```

```julia
# Example

# The resolvent of a random Hermitian matrix is approximately diagonal,
# see Section 3 in https://arxiv.org/pdf/1903.10060.pdf
 normview(resolvent(randHermitian(1000,norm=true))(0.5+0.1im))
```
