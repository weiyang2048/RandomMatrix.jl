module RandomMatrix

using StatsBase: sample, Weights 
import Distributions: mean, var, rand
using Distributions
using Random
using LinearAlgebra
import Polynomials.fit as polyfit 
using SparseArrays: spzeros
using Plots: heatmap, cgrad


const S = Union{Tuple,AbstractArray,Distribution,DataType}

export 
     randDiagonal, randTriangular, randMatrix,
     randHermitian, randSymmetric, GOE, GUE,
     randStochastic,
     randToeplitz, randHankel,
     randUnitary, randOrthogonal , Haar, randPermutation

include("MatrixModels.jl")

include("StructuredMatrices.jl")

include("AdjacencyMatrices.jl")

include("SamplingMatrix.jl")

include("Transformations.jl")

include("Spacing.jl")

include("Distributions.jl")

include("Math.jl")

include("Plots.jl")
end # module
