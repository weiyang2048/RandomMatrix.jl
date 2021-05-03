using Documenter, RandomMatrix

const PAGES = Any["Home" =>"index.md",
      "Random Matrix Models" =>[
            "IID Matrices.md"=>"IID Matrices.md",
            "Others" =>"Others.md"
      ] 

]

makedocs(modules = [RandomMatrix],
      sitename="RandomMatrix.jl",
      authors = "Wei Yang",
      pages = PAGES
      )

deploydocs(repo = "github.com/weiyang2048/RandomMatrix.jl.git",devbranch = "main")