module RandomMatrix

using StatsBase: sample, Weights
import Distributions: mean, var, rand
using Distributions
using Random
using LinearAlgebra
using Statistics: cor
import Polynomials.fit as polyfit
using SparseArrays: spzeros


const S = Union{Tuple,AbstractArray,Distribution,DataType}

export
     randDiagonal, randTriangular, randMatrix,
     randHermitian, randSymmetric, GOE, GUE,
     randStochastic,
     randToeplitz, randHankel,
     randUnitary, randOrthogonal, Haar, COE, CUE, randPermutation

# Plotting entry points. The implementations live in the package extension
# `RandomMatrixPlotsExt` (ext/RandomMatrixPlotsExt.jl) and are only defined
# once the user loads Plots, so Plots is a weak dependency.
export normview, plotEigStats
function normview end
function plotEigStats end

include("MatrixModels.jl")

include("StructuredMatrices.jl")

include("AdjacencyMatrices.jl")

include("SamplingMatrix.jl")

include("Transformations.jl")

include("Spacing.jl")

include("Distributions.jl")

include("Math.jl")

end # module
