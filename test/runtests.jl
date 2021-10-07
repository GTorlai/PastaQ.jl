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
    #"getsamples.jl", the test remotely are extremely slow, skip for now
    "measurements.jl",
    "noise.jl",
    "optimizers.jl",
    "processtomography.jl",
    "productstates.jl",
    "qubitarrays.jl",
    "randomstates.jl",
    "runcircuit.jl",
    "statetomography.jl",
    "utils.jl",
  )
    println("Running $filename")
    include(filename)
  end
end
