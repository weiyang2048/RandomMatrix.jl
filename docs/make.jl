import Pkg: activate
activate(pwd())
cd("docs")


using Documenter, RandomMatrix, Distributions

makedocs(sitename="Documentation")

deploydocs(
    repo = "github.com/weiyang2048/RandomMatrix.jl.git",
)