using Test

@testset "PastaQ.jl" begin
  @testset "$filename" for filename in (
    "gates.jl",
    "circuits.jl",
    "circuitops.jl",
    "runcircuit.jl",
    "getsamples.jl",
    "qubitarrays.jl",
    "randomstates.jl",
    "distances.jl",
    "measurements.jl",
    "optimizers.jl",
    "statetomography.jl",
    "processtomography.jl",
    "utils.jl",
    "examples.jl",
    "observer.jl",
  )
    println("Running $filename")
    include(filename)
  end
end
