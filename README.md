A package for Random Matrix Theory.

Hoping to add-in enough functionalities and register in Aug 2021.
If there is any functionality you want me to implement, please raise an issue.

- [Examples](#examples)
  - [Random Matrix Theory](#random-matrix-theory)
    - [Random Matrix Models](#random-matrix-models)
      - [Random IID Matrices](#random-iid-matrices)
      - [Hermitian Matrices](#hermitian-matrices)
        - [Hermitian](#hermitian)
        - [Symmetric](#symmetric)
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
- `norm` : default is set to `false`, if `norm` set to `true`, then the matrix will be normalized with n^(-1/2).  

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
##### Hermitian
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

##### Symmetric
```julia
randSymmetric(d::T, n::Int; Diag = d::T,  norm = false::Bool) where T<:Union{Distribution{Univariate},DataType,AbstractArray, Tuple}

randSymmetric(n::Int; norm = false::Bool)
```
- Essentially equivalent to `randHermitian` with `complex = false`
- `d` : entry distribution
- `n` : dimensions 
- `norm` : default is set to `false`, if `norm` set to `true`, then the matrix will be normalized with n^(-1/2).  
- `Diag` : the distribution for diagonal entries, by default `Diag=d`. 
    To use a different distribution (say Binomial) for digonal elements, set `Diag = Binomial(1,0.5)`


```julia
# Examples

# Generates a 2 by 2 random Symmetric matrix with entries from the Standard Gaussian.
randSymmetric(2)
``` 

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
using Distributions
pdf(MarchenkoPastur(1.6),0)
```
## Graphics
### Norm View
```julia
normview(M::AbstractMatrix)
```

```julia
# Example

# The resolvent of a random Hermitian matrix is approximately diagonal,
# see Section 3 in https://arxiv.org/pdf/1903.10060.pdf
 normview(resolvent(randHermitian(1000,norm=true))(0.5+0.1im))
```

## Randomized Linear Algebra
### Sampling Matrix
 ```julia
 randSampling(A,B,k=2)
 ```  
>If A is a n by m matrix and B is a m by w matrix.  Run `randSampling(A,B,k=2)` will generate a random  sampling matrix S of size m by k. Where E(SS')=I and 
  that E(ASS'B)=AB.  For definition, check the code or look for definition S:=SD at the end of section 2.2 (right before section 2.3) in https://arxiv.org/pdf/1608.04481.pdf.
