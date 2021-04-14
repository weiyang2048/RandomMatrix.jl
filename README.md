# RandomMatrix.jl

A package for Random Matrix Theory.

Hoping to add-in enough functionalities and publish in Aug 2021.


# Examples

## Random Matrix Theory

### Random Matrix Models
- To generate a 3 by 3 **random Unitary matrix** run 
    - `RandomUnitaryMatrix(3)`  or equivalently
    - `rand(Haar(2,3))`
- To generate a 3 by 3 **random Orthogonal matrix** run 
    - `RandomOrthogonalMatrix(3)` or equivalently
    - `rand(Haar(1,3))`

## Randomized Linear Algebra
- If A is a n by m matrix and B is a m by w matrix.  Run `RandomSamplingMatrix(A,B,k=2)` will generate a random  sampling matrix S. Where E(SS')=I and 
 that E(ASS'B)=AB.  For definition, check the code or look for definition S:=SD at the end of section 2.2 (right before section 2.3) in https://arxiv.org/pdf/1608.04481.pdf.