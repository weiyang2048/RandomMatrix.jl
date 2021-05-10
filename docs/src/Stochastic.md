# Stochastic Matrices
## randStochastic
```@docs
randStochastic
```
## RMT: Circular Law for Doubly Stochastic Random Matrices
Let ``X`` be a matrix sampled uniformly from the set of doubly stochastic matrices of size ``n \times n``. The empirical spectral distribution of the normalized matrix ``\sqrt{n}(X-\mathbf{E} X)`` converges almost surely to the circular law. For reference, see the paper by Hoi H. Nguyen [Random doubly stochastic matrices: The circular law](https://arxiv.org/abs/1205.0843#:~:text=Let%20X%20be%20a%20matrix,surely%20to%20the%20circular%20law.)
```@eval 
using Plots, RandomMatrix, LinearAlgebra
gr() # hide
Plots.reset_defaults() # hide
N = 600
colors = [:red,:green,:blue,:purple]
ani = @animate for n = (1:50...,51:10:N...,N:-10:51...,50:1...)
     
    randStochastic(n,norm=true)|>eigvals|>x->scatter(x,ratio=1,xlims=(-1.5,1.5),title="Circular Law for Stochastic Matrices",size=(600,600),label = "n = $(n)")

    plot!([exp(θ*im) for θ=0:0.01:2pi],label="",lw=3,c=[rand(colors) for _=0:0.01:2pi])
        
end 
gif(ani, "St1.gif", fps = 10); nothing
```
![](St1.gif)