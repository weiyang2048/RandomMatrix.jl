import Pkg: activate
activate(pwd())
cd("docs")


using Documenter, RandomMatrix, Distributions

makedocs(sitename="Documentation")