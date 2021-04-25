module RandomMatrix

using StatsBase: sample, Weights 
import Statistics: mean, var
using Distributions
using Random
using LinearAlgebra 
using SparseArrays: spzeros
using Plots: heatmap, cgrad

include("randMatrix.jl")

include("HermitianMatrix.jl")

include("UnitaryMatrix.jl")

include("SamplingMatrix.jl")

include("Transformations.jl")

include("Distributions.jl")

include("Combinatorics.jl")

include("Plots.jl")
end # module
