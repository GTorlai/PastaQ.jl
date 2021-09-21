using Test

@testset "PastaQ.jl" begin
  @testset "$filename" for filename in (
    "array.jl",
    "utils.jl",
    "gates.jl",
    "noise.jl",
    "circuits.jl",
    "distances.jl",
    "runcircuit.jl",
    "getsamples.jl",
    "optimizers.jl",
    "deprecated.jl",
    "qubitarrays.jl",
    "randomstates.jl",
    "measurements.jl",
    "fulltomography.jl",
    "statetomography.jl",
    "processtomography.jl",
  )
    println("Running $filename")
    include(filename)
  end
end
