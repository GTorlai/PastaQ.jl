include("settings.jl")

## generate the example notebook for the documentation, keep in sync with outputformats.md
#Literate.markdown(joinpath(@__DIR__, "src/outputformats.jl"), OUTPUT; credit = false, name = "name")
#Literate.notebook(joinpath(@__DIR__, "src/outputformats.jl"), OUTPUT; name = "notebook")
#Literate.script(joinpath(@__DIR__, "src/outputformats.jl"), OUTPUT; credit = false)

# Replace the link in outputformats.md
## since that page is not "literated"
#if haskey(ENV, "GITHUB_ACTIONS")
#    folder = Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
#        Documenter.deploy_folder(
#            deployconfig;
#            repo = "github.com/GTorlai/PastaQ.jl.git",
#            devbranch = "master",
#            push_preview = true,
#            devurl = "dev",
#        ).subfolder
#    end
#    url = "https://nbviewer.jupyter.org/github/fredrikekre/Literate.jl/blob/gh-pages/$(folder)/"
#    str = read(joinpath(@__DIR__, "src/outputformats.md"), String)
#    str = replace(str, "[notebook.ipynb](generated/notebook.ipynb)." => "[notebook.ipynb]($(url)generated/notebook.ipynb).")
#    write(joinpath(@__DIR__, "src/outputformats.md"), str)
#end

makedocs(; settings...)

deploydocs(;
  repo = "github.com/GTorlai/PastaQ.jl.git",
  devbranch="master",
  push_preview=true,
  deploy_config=Documenter.GitHubActions(),
)
