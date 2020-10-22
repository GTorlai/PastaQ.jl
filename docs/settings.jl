using Documenter, PastaQ

DocMeta.setdocmeta!(PastaQ,
                    :DocTestSetup,
                    :(using PastaQ);
                    recursive=true)

sitename = "PastaQ.jl"

settings = Dict(
  :modules => [PastaQ],
  :pages => [
             "Introduction" => "index.md",
             "Documentation" => [
                                 "Circuit Simulator" => "Circuits.md",
                                 "Generation of measurement data" => "GetSamples.md",
                                 "Random states" => "RandomStates.md",
                                 "Quantum Tomography" => "QuantumTomography.md",
                                 "Optimizers" => "Optimizers.md",
                                ],
           ],
  :format => Documenter.HTML(assets = ["assets/favicon.ico"],
                             prettyurls = false),
  :doctest => true,
  :checkdocs => :none,
 )
