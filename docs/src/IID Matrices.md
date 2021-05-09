# Matrices with IID Entries

## Diagonal Matrices

```@docs
randDiagonal
```

## TriangularMatrices

```@docs
randTriangular
```

## Full Matrices

```@docs
randMatrix
```


## RMT: Circular Law
Let ``\left(X_{n}\right)_{n=1}^{\infty}`` be a sequence of ``n \times n`` matrix ensembles whose entries are i.i.d. copies of a complex random variable ``x`` with mean ``0`` and variance ``1``. Let ``\lambda_{1}, \ldots, \lambda_{n}, 1 \leq j \leq n`` denote the eigenvalues of ``\frac{1}{\sqrt{n}} X_{n}``. Define the empirical spectral measure of ``\frac{1}{\sqrt{n}} X_{n}`` as
```math
\mu_{\frac{1}{\sqrt{n}} X_{n}}(A)=n^{-1} \#\left\{j \leq n: \lambda_{j} \in A\right\}, \quad A \in \mathcal{B}(\mathbb{C})
```
The circular law asserts that almost surely (i.e. with probability one), the sequence of measures ``\mu \frac{1}{\sqrt{n}} X_{n}`` converges in distribution to the uniform measure on the unit disk.

```@eval 
using Plots, RandomMatrix, LinearAlgebra
gr() # hide
Plots.reset_defaults() # hide
N = 500
M = randMatrix(N)
colors = [:red,:green,:blue,:purple]
anim = @animate for n = (1:50...,51:10:N...,N:-10:51...,50:1...)
     
    M[1:n,1:n]/sqrt(n)|>eigvals|>x->scatter(x,ylims=(-1.25,1.25), xlims=(-1.25,1.25),ratio=1,label="n = $(n)",size=(600,600))

    plot!([exp(θ*im) for θ=0:0.01:2pi],label="",lw=3,c=[rand(colors) for _=0:0.01:2pi])
        
    title!("Circular Law for IID Matrices")
    
end 
gif(anim, "IID1.gif", fps = 10)
```
![](IID1.gif)