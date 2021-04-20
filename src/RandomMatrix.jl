module RandomMatrix

using StatsBase: sample, Weights 
using Distributions
using Random: shuffle
using LinearAlgebra 
using SparseArrays: spzeros
using Plots: heatmap, cgrad

include("HermitianMatrix.jl")

include("UnitaryMatrix.jl")

include("SamplingMatrix.jl")

include("Transformations.jl")

include("Distributions.jl")

include("Combinatorics.jl")

include("Plots.jl")
end # module
