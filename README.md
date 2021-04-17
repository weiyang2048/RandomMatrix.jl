# RandomMatrix.jl

A package for Random Matrix Theory.

Hoping to add-in enough functionalities and publish in Aug 2021.


# Examples

## Random Matrix Theory

### Random Matrix Models
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
```julia
randPermutation(n)
```
> Generates a  n by n **random permutation matrix**
#### Hermitian Matrices
```julia
randHermitian(2)
```  
>Generates a 2 by 2 random Symmetric/Hermitian matrix with entries from the real Gaussian(0,1).
```julia
randHermitian(ComplexF64,2)
```  
>Generates a 2 by 2 random Hermitian matrix with entries from the standard complex Gaussian.
```julia
randHermitian(Poisson(2),2)
```   
>Generate a random 2 by 2 Hermitian Matrix with entries  `Poisson(2)` rvs.  
*Need to import the `Distributions` package for `Poisson(2)`*

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