export Resolvent, qresolvent

abstract type MatrixTransformations end

"""
```julia
Resolvent(A::Matrix)
```

- return the resolvent **function** of `A`, ``R(z)= (A-zI)^{-1}``

# Examples

Get the resolvent **function** of a normalized GOE(500).

And evaluate the resolvent at `0`
```julia
R = Resolvent(rand(GOE(500),norm = true)) # this a function

R(0)

500×500 Matrix{Float64}:
  0.460903   -0.384016   0.268931  …  -0.59029    -0.178288
 -0.384016    0.489355  -0.366855      0.452094   -0.190179
  0.268931   -0.366855  -0.697169     -0.221433    0.535448
  0.388846    0.946668  -0.354459     -0.796717    0.207156
  ⋮                                ⋱
 -0.532533    0.950826  -0.366618      0.0926931  -0.288692
 -0.0874979  -0.553262  -0.388048     -0.0156897   0.177695
 -0.59029     0.452094  -0.221433      0.9136      0.343194
 -0.178288   -0.190179   0.535448      0.343194   -0.141701
```
"""
struct Resolvent<:MatrixTransformations
    M::Matrix
end


function (Resolvent::MatrixTransformations)(z)
    (Resolvent.M-UniformScaling(z))^(-1)
end

function Base.show(io::IO, R::Resolvent) 
    print("Resolvent{Matrix}")
end

"""
```julia
qresolvent(A)
```
returns the quaternion resolvent **function** of A.
"""
function qresolvent(A::AbstractMatrix)
    return function f(z::Number,η::Number)
    ([UniformScaling(-η) A-UniformScaling(z) 
    A'-UniformScaling(z') UniformScaling(-η)])^(-1)
end
end