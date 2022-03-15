using Documenter
using PastaQ
using Literate
using Plots 

DocMeta.setdocmeta!(PastaQ, :DocTestSetup, :(using PastaQ); recursive=true)

INPUT_SRC_DIR = joinpath(@__DIR__, "..", "examples", "src")
OUTPUT_JL_DIR = joinpath(@__DIR__, "..", "examples", "scripts")
OUTPUT_NB_DIR = joinpath(@__DIR__, "..", "examples", "notebooks")
OUTPUT_MD_DIR = joinpath(@__DIR__, "src","examples")

#for file in filter(endswith(file, en), readdir(EXAMPLES_SRC_DIR))

examples_files = filter(x -> endswith(x, ".jl"), readdir(INPUT_SRC_DIR))
for file in examples_files
  EXAMPLE = joinpath(INPUT_SRC_DIR, file)
  Literate.markdown(EXAMPLE, OUTPUT_MD_DIR)
  Literate.script(EXAMPLE, OUTPUT_JL_DIR)
  Literate.notebook(EXAMPLE, OUTPUT_NB_DIR)
end

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

makedocs(
    modules = [PastaQ],
    sitename = "PastaQ.jl",
    pages = Any[
      "Home" => "index.md",
      "Documentation" => [
        "Quantum states"          => "quantumstates.md",
        "Quantum circuits"        => "quantumcircuits.md",
        "Measurements"            => "measurements.md",
        "Quantum tomography"      => "quantumtomography.md",
        "Quantum dynamics"        => "trottercircuits.md",
      ],
      "Tutorials" => [
        "Getting started"                    => "examples/getting-started.md",
        #"Quantum Fourier transform"          => "examples/getting-started.md",
        #"Simulated XEB experiment"           => "examples/getting-started.md",
        #"Monitored quantum circuits"         => "examples/getting-started.md",
        #"Trotter dynamics"                   => "examples/getting-started.md",
        "Optimal coherent control"           => "examples/optimal-coherent-control.md",
        #"Variational quantum eigensolver"    => "examples/getting-started.md",
        #"Many-body quantum state tomography" => "examples/getting-started.md",
        #"Noise characterization"             => "examples/getting-started.md",
      ]
    ],
    format = Documenter.HTML(; assets=["assets/favicon.ico"], prettyurls=false),
    doctest = false,
    checkdocs = :none
)

