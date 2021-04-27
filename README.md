# RandomMatrix.jl

A package for Random Matrix Theory.

Hoping to add-in enough functionalities and register in Aug 2021.
If there is any functionality you want me to implement, please raise an issue.

- [RandomMatrix.jl](#randommatrixjl)
- [Examples](#examples)
  - [Random Matrix Theory](#random-matrix-theory)
    - [Random Matrix Models](#random-matrix-models)
      - [Random IID Matrices](#random-iid-matrices)
      - [Hermitian Matrices](#hermitian-matrices)
      - [Unitary Matrices](#unitary-matrices)
    - [Transformations](#transformations)
      - [Resolvent](#resolvent)
      - [Quaternion Resolvent](#quaternion-resolvent)
  - [Distributions](#distributions)
    - [Complex Gaussian](#complex-gaussian)
      - [Gaussian](#gaussian)
    - [Circular Law](#circular-law)
    - [Elliptical Law](#elliptical-law)
    - [Marchenko-Pastur Law](#marchenko-pastur-law)
  - [Graphics](#graphics)
    - [Norm View](#norm-view)
  - [Randomized Linear Algebra](#randomized-linear-algebra)
    - [Sampling Matrix](#sampling-matrix)

# Examples

## Random Matrix Theory

### Random Matrix Models

#### Random IID Matrices
```julia
randMatrix(d::T, n::Int, m = n::Int  ; norm = false::Bool) where T<:Union{Distribution{Univariate},DataType,AbstractArray, Tuple}

randMatrix(n::Int, m = n :: Int; norm = false::Bool)
```
- `d` : entry distribution
- `n` , `m` : dimensions, if `m` is not provided, by default `m=n` 
- `norm` : default is set to `false`, if `norm` set to `true`, then the matrix will be normlaized with n^(-1/2).  

```julia
# Examples

# Generates a 2 by 2 random  matrix with entries from the Standard  Gaussian.
randMatrix(2)

# Generates a 3 by 2 random  matrix with entries uniformly from {1,2,3,...,10}.
randMatrix(1:10,3,2)

# Generate a normalized random 2 by 2  Matrix with entries  `Poisson(2)` rvs. 
# Need to import the `Distributions` package for `Poisson(2)`
randMatrix(Poisson(2),2,norm = true)
``` 
#### Hermitian Matrices
```julia
randHermitian(d::T, n::Int; Diag = d, norm = false::Bool, complex=true::Bool) where T<:Union{Distribution{Univariate},DataType,AbstractArray, Tuple}

randHermitian(n::Int; norm = false::Bool)
```
- `d` : entry distribution
- `n`  : dimensions 
- `norm` : default is set to `false`, if `norm` set to `true`, then the matrix will be normlaized with n^(-1/2).  
- `Diag` : the distribution for diagonal entries, by default `Diag=d`. 
    To use a different distribution (say Binomial) for digonal elements, set `Diag = Binomial(1,0.5)`
- `complex` : by default `complex = true`, we assume entries will be complex, if one knows that all entries will be real, set `complex=false`,
    or equivalently use `randSymmetric`


```julia
# Examples

# Generates a 2 by 2 random Hermitian matrix with entries from the Standard Complex Gaussian.
randHermitian(2)

# Generate a random 2 by 2 Symmetric Matrix with entries  `Poisson(2)` rvs. 
# *Need to import the `Distributions` package for `Poisson(2)`*
randHermitian(Poisson(2),2)
# or equivalently 
randSymmetric(Poisson(2),2)

# Entries uniformly from {1,2,3,...,10}
randHermitian(1:10,2)

# Entries either -1 or pi with equal probability
randHermitian([-1,pi],2)
``` 
```julia
randSymmetric(d::T, n::Int; Diag = d::T,  norm = false::Bool) where T<:Union{Distribution{Univariate},DataType,AbstractArray, Tuple}

randSymmetric(n::Int; norm = false::Bool)
```
- Essentially equivalent to `randHermitian` with `complex = false`
- To use a different distribution (say Binomial) for digonal elements, set `Diag = Binomial(1,0.5)`
- If `norm` set to `true`, then the matrix will be normalized with n^(-1/2).
***  
```julia
randSymmetric(2)
``` 
>Generates a 2 by 2 random Symmetric matrix with entries from the Standard Gaussian.

***
***
#### Unitary Matrices 
```julia
randUnitary(3)
# or
rand(Haar(2,3))
``` 
>Generates a 3 by 3 **random Unitary matrix** 
```julia
randOrthogonal(3)
# or
rand(Haar(1,3))
```
>Generates a 3 by 3 **random Orthogonal matrix**
***
```julia
randPermutation(n::Int, fix = 0::Int) 
```
>Generate a random permutation matrix.  If `fix =x`, 
>`randPermutation(n,x)` will have at-least `x` fixed points.
***
```julia
randPermutation(n) # fun fact, trace of a large random Permutation matrix is a Poisson(1) rv.
```
> Generates a  n by n **random permutation matrix**
***
***

### Transformations
#### Resolvent
```julia
resolvent(A)
```
>return the resolvent ***function*** of A
#### Quaternion Resolvent
```julia
qresolvent(A)
```
>returns the quaternion resolvent ***function*** of A.

## Distributions
### Complex Gaussian
```julia
ComplexNormal
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
Gaussian{Int8,ComplexF64,Float64} 
````
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
MarchenkoPastur(λ::Float64,σ::Float64)
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
## Graphics
### Norm View
```julia
normview(M::AbstractMatrix)
```

```julia
# Example

# The resolvent of a random Hermitian matrix is approximately diagonal, see Section 3 in https://arxiv.org/pdf/1903.10060.pdf
 normview(resolvent(randHermitian(1000,norm=true))(0.5+0.1im))
```

## Randomized Linear Algebra
### Sampling Matrix
 ```julia
 randSampling(A,B,k=2)
 ```  
>If A is a n by m matrix and B is a m by w matrix.  Run `randSampling(A,B,k=2)` will generate a random  sampling matrix S of size m by k. Where E(SS')=I and 
  that E(ASS'B)=AB.  For definition, check the code or look for definition S:=SD at the end of section 2.2 (right before section 2.3) in https://arxiv.org/pdf/1608.04481.pdf.
