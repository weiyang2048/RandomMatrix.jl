# Stochastic Matrices
## randStochastic
```@docs
randStochastic
```
## Some Random Matrix Theory
```@example 
using Plots, RandomMatrix, LinearAlgebra
gr() # hide
Plots.reset_defaults() # hide
N = 500
colors = [:red,:green,:blue,:purple]
@gif for n = (1:50...,51:10:N...,N:-10:51...,50:1...)
     
    randStochastic(n,norm=true)|>eigvals|>x->scatter(x,ratio=1,xlims=(-1.5,1.5),title="Circular Law for Stochastic Matrices",size=(600,600),label = "n = $(n)")

    plot!([exp(θ*im) for θ=0:0.01:2pi],label="",lw=3,c=[rand(colors) for _=0:0.01:2pi])
        
end 
```