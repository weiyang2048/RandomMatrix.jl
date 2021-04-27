module RandomMatrix

using StatsBase: sample, Weights 
import Distributions: mean, var
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

include("Math.jl")

include("Plots.jl")
end # module
