export powerset, ±

"""
Generate the powerset of 
```julia
julia> powerset([1,2,3])
[[1], [2], [1, 2],[3],[1, 3],[2, 3]]
```
"""
function powerset(x::AbstractArray)   
    result = [[]]
    for elem in x, j in eachindex(result)
        push!(result, [result[j] ; elem])
    end
    result
end


"""
```julia
±(a,b)
```
- returns (a-b,a+b)
```julia
# Examples

1 ± 0.5  # returns (0.5,1.5)
```
"""
±(a,b) = [a-b,a+b]