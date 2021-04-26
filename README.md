# RandomMatrix.jl

A package for Random Matrix Theory.

Hoping to add-in enough functionalities and publish in Aug 2021.

Note that there is an existing package [`RandomMatrices.jl`](https://github.com/JuliaMath/RandomMatrices.jl) on Random Matrix, but it is no longer maintained (the lastest version is released in 2019 Julia 1.1 era, some of the codes no longer work), and it doesn't provide all the functionalities that I need in my work.  Thus it probably will be easier to start my own package.  If there is any functionality you want me to implement, please raise an issue.

- [RandomMatrix.jl](#randommatrixjl)
- [Examples](#examples)
  - [Random Matrix Theory](#random-matrix-theory)
    - [Random Matrix Models](#random-matrix-models)
      - [Random IID Matrices](#random-iid-matrices)
      - [Hermitian Matrices](#hermitian-matrices)
      - [Unitary Matrices](#unitary-matrices)
    - [Transformations](#transformations)
      - [Resolvent](#resolvent)
  - [Randomized Linear Algebra](#randomized-linear-algebra)
  - [Probability](#probability)
    - [Distributions](#distributions)
      - [Complex Gaussian](#complex-gaussian)
      - [Circular Distribution](#circular-distribution)

# Examples

## Random Matrix Theory

### Random Matrix Models

#### Random IID Matrices

```julia
randMatrix(d::T, n::Int, m = n::Int; norm = false::Bool) where T<:Union{Distribution{Univariate},DataType,AbstractArray, Tuple}

randMatrix(n::Int, m = n ::Int; norm = false::Bool)
```
- If `norm` set to `true`, then the matrix will be normlaized with n^(-1/2).  

```julia
randMatrix(2)
``` 
>Generates a 2 by 2 random matrix with entries from the Standard  Gaussian.
```julia
randMatrix(1:10,3,2)
``` 
>Generates a 3 by 2 random  matrix with entries uniform in {1,2,3,...,10}.

```julia
randMatrix(Poisson(2), 2, norm = true)
``` 

>Generate a normalized random 2 by 2  Matrix with entries  `Poisson(2)` rvs. 
>*Need to import the `Distributions` package for `Poisson(2)`*

***
***
#### Hermitian Matrices
***
```julia
randHermitian(d::T, n::Int; Diag=d::T, norm = false::Bool, complex=true::Bool) where T<:Union{Distribution{Univariate},DataType,AbstractArray, Tuple}

randHermitian(n::Int; norm = false::Bool)
```
- To use a different distribution (say Binomial) for digonal elements, set `Diag = Binomial(1,0.5)`
- If `norm` set to `true`, then the matrix will be normlaized with n^(-1/2).  
- If one knows that all entries will be real, set `complex=false`,
    or equivalently use `randSymmetric`
***
```julia
randHermitian(2)
``` 
>Generates a 2 by 2 random Hermitian matrix with entries from the Standard Complex Gaussian.
```julia
randHermitian(Poisson(2),2)
``` 
or
```julia
randSymmetric(Poisson(2),2)
```
>Generate a random 2 by 2 Symmetric Matrix with entries  `Poisson(2)` rvs. 
>*Need to import the `Distributions` package for `Poisson(2)`*
```julia
randHermitian(1:10,2)
``` 
```julia
randHermitian([-1,pi],2)
```
***
```julia
randSymmetric(d::T, n::Int; Diag = d::T,  norm = false::Bool) where T<:Union{Distribution{Univariate},DataType,AbstractArray, Tuple}

randSymmetric(n::Int; norm = false::Bool)
```
- Essentially equivalent to `randHermitian` with `complex = false`
- To use a different distribution (say Binomial) for digonal elements, set `Diag = Binomial(1,0.5)`
- If `norm` set to `true`, then the matrix will be normlaized with n^(-1/2).
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
```julia
qresolvent(A)
```
>returns the quaternion resolvent ***function*** of A.
## Randomized Linear Algebra
 ```julia
 randSampling(A,B,k=2)
 ```  
>If A is a n by m matrix and B is a m by w matrix.  Run `randSampling(A,B,k=2)` will generate a random  sampling matrix S of size m by k. Where E(SS')=I and 
  that E(ASS'B)=AB.  For definition, check the code or look for definition S:=SD at the end of section 2.2 (right before section 2.3) in https://arxiv.org/pdf/1608.04481.pdf.

## Probability
### Distributions
#### Complex Gaussian
***
```julia
Gaussian(beta=1,μ=0,σ=1)
````
- `beta`: 1 for Real Gaussian, 2 for Complex Gaussian 
***
```julia
ComplexNormal(μ=0,σ=1) 
```
***
#### Circular Distribution
```julia
Circular(c=0,R=1)
```
- The uniform distribution on the complex disk with center `c` and radius `R`
***

```julia
rand(Circular(1,10),200)
```
> Generates 200 uniform r.v.s on the complex disk of center 1 radius 10.