 
```@example
using Plots, RandomMatrix, LinearAlgebra, Distributions, Plots.PlotMeasures 
gr()  # hide
Plots.reset_defaults() # hide
N = 1000
M = randMatrix(N)
H = randHermitian(N)
U = randUnitary(N)
l=@layout[c c; c c]
colors = [:red,:green,:blue,:purple]
 
@gif for n = (1:50...,51:10:N...,N:-10:51...,50:1...)
     
  p1 = M[1:n,1:n]/sqrt(n)|>eigvals|>x->scatter(x,ylims=(-1.25,1.25), xlims=(-1.25,1.25),ratio=1,label="n = $(n)")
        plot!([exp(θ*im) for θ=0:0.01:2pi],label="",lw=3,c=[rand(colors) for _=0:0.01:2pi])
        title!("Circular Law for IID Matrices")
    p3 = U[1:N-n÷2,1:N-n÷2]|>eigvals|>x->scatter(x,ylims=(-1.25,1.25), xlims=(-1.25,1.25),ratio=1,label="ratio = $(round((N-n÷3)/N,digits=2))")
        plot!([exp(θ*im) for θ=0:0.01:2pi],label="",lw=3,c=[rand(colors) for _=0:0.01:2pi])
        plot!(sqrt((N-n÷2)/N)*[exp(θ*im) for θ=0:0.01:2pi],label="",lw=3,c=[rand(colors) for _=0:0.01:2pi],alpha=0.1)
        title!("Circular Law for Truncated Unitary")
  p2 = H[1:n,1:n]/sqrt(n)|> eigvals|> x->histogram(x,norm=true, ylims=(0,1/pi+0.2), xlims=(-2.25,2.25),nbins = ceil(Int,sqrt(n)),label="n = $(n)")
        plot!(x->pdf(Semicircle(2),x),-2:0.01:2,lw=3,label="",c=[rand(colors) for _=-2:0.01:2])
        title!("Semicircle Law for Hermitian Matrices")
            p4 = randEllipic(500,r=((N-2*n)/N)+0.0001,norm=true)|>eigvals|>x->scatter(x,ylims=(-2,2), xlims=(-2,2),ratio=1,label="ρ = $(round(((N-2*n)/N)+0.0001,digits=2))")
   title!("Elliptical Law")
  plot(p1,p2,p3,p4,size = (1000, 1000),axis=false)
end 
```



 
```@example
using Plots, RandomMatrix, LinearAlgebra, Distributions
gr()  # hide
Plots.reset_defaults() # hide
N = 500
M = randMatrix(N)
U = randUnitary(N)
l=@layout[c c]
colors = [:red,:green,:blue,:purple]
 
@gif for n = (1:50...,51:10:N...,N:-10:51...,50:1...)
     
  p1 = M[1:n,1:n]/sqrt(n)|>eigvals|>x->scatter(x,ylims=(-1.25,1.25), xlims=(-1.25,1.25),ratio=1,label="n = $(n)")
        plot!([exp(θ*im) for θ=0:0.01:2pi],label="",lw=3,c=[rand(colors) for _=0:0.01:2pi])
        title!("Circular Law for IID Matrices")
    p2 = U[1:N-n÷2,1:N-n÷2]|>eigvals|>x->scatter(x,ylims=(-1.25,1.25), xlims=(-1.25,1.25),ratio=1,label="ratio = $(round((N-n÷3)/N,digits=2))")
        plot!([exp(θ*im) for θ=0:0.01:2pi],label="",lw=3,c=[rand(colors) for _=0:0.01:2pi])
        plot!(sqrt((N-n÷2)/N)*[exp(θ*im) for θ=0:0.01:2pi],label="",lw=3,c=[rand(colors) for _=0:0.01:2pi],alpha=0.1)
        title!("Circular Law for Truncated Unitary")

  plot(p1,p2,size = (1000, 500),axis=false)
end 
```
