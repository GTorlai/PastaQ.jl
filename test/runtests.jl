using Test

@testset "PastaQ.jl" begin
  @testset "$filename" for filename in (
    "utils.jl",
    "quantumgates.jl",
    "circuits.jl",
    "circuitops.jl",
    "quantumcircuit.jl",
    "statetomography.jl"
  )
    println("Running $filename")
    include(filename)
  end
end