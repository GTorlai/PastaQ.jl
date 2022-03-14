using Documenter, PastaQ

DocMeta.setdocmeta!(PastaQ, :DocTestSetup, :(using PastaQ); recursive=true)

sitename = "PastaQ.jl"

settings = Dict(
  :modules => [PastaQ],
  :pages => [
    "Home" => "index.md",
    "Documentation" => [
      "Quantum states" => "quantumstates.md",
      "Quantum circuits" => "quantumcircuits.md",
      "Noise" => "noise.md",
      "Measurements" => "measurements.md",
      "Quantum dynamics" => "trottercircuits.md",
      "Differentiable circuits" => "differentiablecircuits.md",
      "Quantum tomography" => "quantumtomography.md",
    ],
    "Tutorials" => [
      "Getting started" => "examples/example1.md",
      "Quantum Fourier transform" => "examples/example1.md",
      "Simulated XEB experiment" => "examples/example1.md",
      "Monitored quantum circuits" => "examples/example1.md",
      "Trotter dynamics" => "examples/example1.md",
      "Optimal coherent control" => "examples/example1.md",
      "Variational quantum eigensolver" => "examples/example1.md",
      "Many-body quantum state tomography" => "examples/example1.md",
      "Noise characterization" => "examples/example1.md",
    ]
  ],
  :format => Documenter.HTML(; assets=["assets/favicon.ico"], prettyurls=false),
  :doctest => true,
  :checkdocs => :none,
)
