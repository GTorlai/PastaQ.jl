using Documenter
using PastaQ

DocMeta.setdocmeta!(PastaQ, :DocTestSetup, :(using PastaQ); recursive=true)

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
  ],
  :format => Documenter.HTML(; assets=["assets/favicon.ico"], prettyurls=false),
  :doctest => true,
  :checkdocs => :none
)
