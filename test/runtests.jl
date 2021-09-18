using Test

@testset "PastaQ.jl" begin
  @testset "$filename" for filename in (
    "array.jl",
    "gates.jl",
    "noise.jl",
    "circuits.jl",
    "circuitops.jl",
    "runcircuit.jl",
    "getsamples.jl",
    "qubitarrays.jl",
    "randomstates.jl",
    "distances.jl",
    "measurements.jl",
    "statetomography.jl",
    "processtomography.jl",
    "utils.jl",
    "observer.jl",
  )
    println("Running $filename")
    include(filename)
  end
end
