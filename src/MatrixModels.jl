const MatrixModels = ["IID Matrices", 
                "Hermitian Matrices",
                "Unitary Matrices",
                "CorrelatedMatrices"]

for model in MatrixModels
    include(joinpath("MatrixModels", "$(model).jl"))
end