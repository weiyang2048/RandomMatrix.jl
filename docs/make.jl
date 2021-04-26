import Pkg: activate
activate(pwd())
cd("docs")


using Documenter, RandomMatrix

makedocs(sitename="Documentation")