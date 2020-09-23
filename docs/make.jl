include("settings.jl")

makedocs(sitename = sitename; settings...)

deploydocs(repo = "github.com/GTorlai/PastaQ.jl")
