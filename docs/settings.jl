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
                                 "Circuits" => "Circuits.md",
                                ],
           ],
  :format => Documenter.HTML(assets = ["assets/favicon.ico"],
                             prettyurls = false),
  :doctest => true,
  :checkdocs => :none,
 )
