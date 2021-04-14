module RandomMatrix

using LinearAlgebra, StatsBase, Distributions

include("Unitary.jl")

export RandomSamplingMatrix


"""
```julia
RandomSamplingMatrix(A::Matrix, B=I::Matrix; k=0::Int)
```
Generate a random sampling random Matrix `S`, E(SS')=I.
ASS'B is typically used to approximate AB.


# Examples
```julia
S = RandomSamplingMatrix(rand(2,3),rand(3,2),k=2)

3×2 Matrix{Float64}:
 0.0      0.0
 0.0      1.2439
 1.15342  0.0
```
"""
function RandomSamplingMatrix(A::Matrix,B=I::Matrix; k=0::Int)
    if k <= 0 
        k = floor(Int,size(A)[2]/2)
        print("k was <= $(0) or not provided, it is now set to $(k) by default")
    end
    
    n, m = size(A)
    if size(B)[1] != m
        return "The matrices can not be multiplied "
    else
        Lens = [norm(A[:,i])*norm(B[i,:]) for i = 1:m ]
        Total = sum(Lens)
        S = zeros(m,k)
        for (i,j) in enumerate([sample([1:m...], Weights(Lens) ) for i = 1 : k])
            S[j,i] = 1/sqrt(Lens[j]/Total*k) # normalizing so that E[SS']=I
        end
        return S
    end
end


end # module
