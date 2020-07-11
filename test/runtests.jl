using Test

@testset "PastaQ.jl" begin
  @testset "$filename" for filename in (
    "ITensorsGateEvolution/runtests.jl",
    "utils.jl",
    "quantumgates.jl",
    "circuitops.jl",
    "quantumcircuit.jl",
    "statetomography.jl"
  )
    println("Running $filename")
    include(filename)
  end
end

