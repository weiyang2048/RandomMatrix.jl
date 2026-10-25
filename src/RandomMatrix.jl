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

"""
    normview(M::AbstractMatrix)

Heatmap of the entrywise norms of `M`. Requires Plots to be loaded
(provided by the `RandomMatrixPlotsExt` package extension).
"""
function normview end

"""
    plotEigStats(M::AbstractMatrix)

Plot eigenvalue statistics of `M`: eigenvalue histogram, unfolded-spacing
histogram against the Wigner surmise, and the number variance. Requires
Plots to be loaded (provided by the `RandomMatrixPlotsExt` package
extension).
"""
function plotEigStats end

include("MatrixModels.jl")

include("StructuredMatrices.jl")

include("AdjacencyMatrices.jl")

include("SamplingMatrix.jl")

include("Transformations.jl")

include("SpikedCovariance.jl")

include("CovarianceCleaning.jl")

include("TracyWidom.jl")

include("FreeProbability.jl")

include("Spacing.jl")

include("Distributions.jl")

include("Math.jl")

end # module
