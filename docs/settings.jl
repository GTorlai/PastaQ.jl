using Documenter
using PastaQ

DocMeta.setdocmeta!(PastaQ, :DocTestSetup, :(using PastaQ); recursive=true)
using Literate
using Plots # to not capture precompilation output

INPUT_SRC_DIR = joinpath(@__DIR__, "..", "examples", "src")
OUTPUT_JL_DIR = joinpath(@__DIR__, "..", "examples", "scripts")
OUTPUT_NB_DIR = joinpath(@__DIR__, "..", "examples", "notebooks")
OUTPUT_MD_DIR = joinpath(@__DIR__, "src","examples")

examples_files = filter(x -> endswith(x, ".jl"), readdir(INPUT_SRC_DIR))
for file in examples_files
  EXAMPLE = joinpath(INPUT_SRC_DIR, file)
  Literate.markdown(EXAMPLE, OUTPUT_MD_DIR)
  Literate.script(EXAMPLE, OUTPUT_JL_DIR)
  Literate.notebook(EXAMPLE, OUTPUT_NB_DIR)
end

settings = Dict(
  :modules => [PastaQ],
  :sitename => "PastaQ.jl",
  :pages => Any[
    "Home" => "index.md",
    "Documentation" => [
      "Quantum states"      => "quantumstates.md",
      "Quantum circuits"    => "quantumcircuits.md",
      "Measurements"        => "measurements.md",
      "Quantum tomography"  => "quantumtomography.md",
    ],
    "Tutorials" => [
      "Optimal coherent control" => "examples/optimal-coherent-control.md" 
    ]
  ],
  :format => Documenter.HTML(; assets=["assets/favicon.ico"], prettyurls=false),
  :doctest => true,
  :checkdocs => :none
)



