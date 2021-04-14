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
- If $A\in R^{n,m}$ and $B\in R^{m,w}$, run `RandomSamplingMatrix(A,B,k=2)` will generate a random  sampling matrix $\mathcal{S}\in R^{m,k}$. Note that $\mathbf{E}(\mathcal{S}\mathcal{S}')=I$ and 
 that $\mathbf{E}(A\mathcal{S}\mathcal{S}'B)=AB$.  For defefinition, check the code or look for definition at the end of section 2.2 in https://arxiv.org/pdf/1608.04481.pdf.