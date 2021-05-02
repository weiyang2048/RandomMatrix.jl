using Documenter, RandomMatrix

makedocs(modules = [RandomMatrix],
      sitename="RandomMatrix.jl")

deploydocs(repo = "github.com/weiyang2048/RandomMatrix.jl.git",devbranch = "main")