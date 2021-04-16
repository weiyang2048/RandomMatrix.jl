module RandomMatrix

using LinearAlgebra, StatsBase, Distributions

include("HermitianMatrix.jl")

include("UnitaryMatrix.jl")

include("SamplingMatrix.jl")

include("Transformations.jl")

end # module
