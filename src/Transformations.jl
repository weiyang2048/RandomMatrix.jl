export resolvent, qresolvent

"""
```julia
resolvent(A)
```
return the resolvent ***function*** of A
"""
function resolvent(A::AbstractMatrix)
    return function f(z::Number)
    (A-UniformScaling(z))^(-1)
end
end

"""
```julia
qresolvent(A)
```
returns the quaternion resolvent ***function*** of A.
"""
function qresolvent(A::AbstractMatrix)
    return function f(z::Number,η::Number)
    ([UniformScaling(-η) A-UniformScaling(z) 
    A'-UniformScaling(z') UniformScaling(-η)])^(-1)
end
end