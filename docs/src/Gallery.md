 
```@eval

using Plots, RandomMatrix, LinearAlgebra, Distributions
gr()  
Plots.reset_defaults() 
N = 600
M = randMatrix(N)
U = randUnitary(N)
l=@layout[c c; c c]
colors = [:red,:green,:blue,:purple]
 
anim = @animate for n = (1:50...,51:10:N...,N:-10:51...,50:1...)
     
  p1 = M[1:n,1:n]/sqrt(n)|>eigvals|>x->scatter(x,ylims=(-1.25,1.25), xlims=(-1.25,1.25),ratio=1,label="n = $(n)")
        plot!([exp(θ*im) for θ=0:0.01:2pi],label="",lw=3,c=[rand(colors) for _=0:0.01:2pi])
        title!("Circular Law for IID Matrices")
    
    p2 = U[1:N-n÷2,1:N-n÷2]|>eigvals|>x->scatter(x,ylims=(-1.25,1.25), xlims=(-1.25,1.25),ratio=1,label="ratio = $(round((N-n÷3)/N,digits=2))")
        plot!([exp(θ*im) for θ=0:0.01:2pi],label="",lw=3,c=[rand(colors) for _=0:0.01:2pi])
        plot!(sqrt((N-n÷2)/N)*[exp(θ*im) for θ=0:0.01:2pi],label="",lw=3,c=[rand(colors) for _=0:0.01:2pi],alpha=0.1)
        title!("Circular Law for Truncated Unitary")

    p3 = randStochastic(n,norm=true)|>eigvals|>x->scatter(x,ratio=1,xlims=(-1.5,1.5),title="Circular Law for Stochastic Matrices",size=(600,600),label = "n = $(n)")

    plot!([exp(θ*im) for θ=0:0.01:2pi],label="",lw=3,c=[rand(colors) for _=0:0.01:2pi])
    
    p4 = randEllipic(500,r=((N-2*n)/N)+0.0001,norm=true)|>eigvals|>x->scatter(x,ylims=(-2,2), xlims=(-2,2),ratio=1,label="ρ = $(round(((N-2*n)/N)+0.0001,digits=2))")
   title!("Elliptical Law")
  plot(p1,p2,p3,p4,size = (1000, 1000),axis=false)
end 
gif(anim, "testing1.gif", fps = 10)
; nothing
```



 ![](testing1.gif)