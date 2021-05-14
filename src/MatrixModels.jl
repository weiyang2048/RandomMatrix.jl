const MatrixModels = ["IID Matrices", 
                "Hermitian Matrices",
                "Unitary Matrices"]

for model in MatrixModels
    include(joinpath("MatrixModels", "$(model).jl"))
end