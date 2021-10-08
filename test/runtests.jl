using Test

@testset "PastaQ.jl" begin
  @testset "$filename" for filename in (
    "array.jl",
    "circuits.jl",
    "distances.jl",
    "deprecated.jl",
    "fulltomography.jl",
    "gates.jl",
    "io.jl",
    "getsamples.jl", 
    "measurements.jl",
    "noise.jl",
    "optimizers.jl",
    "processtomography.jl",
    "productstates.jl",
    "qubitarrays.jl",
    "randomstates.jl",
    "runcircuit.jl",
    "statetomography.jl",
    "trottersuzuki.jl",
    "utils.jl",
  )
    println("Running $filename")
    include(filename)
  end
end
