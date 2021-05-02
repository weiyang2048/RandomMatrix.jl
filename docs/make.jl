using Documenter, RandomMatrix

makedocs(sitename="RandomMatrix.jl")

deploydocs(
    repo = "github.com/weiyang2048/RandomMatrix.jl.git",
    target = "build",
    branch = "gh-pages",
)