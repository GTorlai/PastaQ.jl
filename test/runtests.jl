using Test

@testset "PastaQ.jl" begin
  @testset "$filename" for filename in (
    "gates.jl",
    "noise.jl",
    "circuits.jl",
    "runcircuit.jl",
    "getsamples.jl",
    "qubitarrays.jl",
    "randomstates.jl",
    "distances.jl",
    "measurements.jl",
    "fulltomography.jl",
    "statetomography.jl",
    "processtomography.jl",
    "array.jl",
    "utils.jl",
  )
    println("Running $filename")
    include(filename)
  end
end
