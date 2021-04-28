export resolvent, qresolvent

"""
```julia
resolvent(A::Matrix)
```

- return the resolvent ***function*** of `A`, ``R(z)= (A-zI)^{-1}``

```julia
# Examples

# This returns the resolvent **function** of a 500 by 500 Hermitian 
resolvent(randHermitian(500, norm = true)) # this is a generic function

# One can use the above return as a function
f = resolvent(randHermitian(500, norm = true))
f(0+0.1im) # this returns a matrix
```
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