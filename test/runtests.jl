using Test

@testset "PastaQ.jl" begin
  @testset "$filename" for filename in (
    "array.jl",
    "autodiff.jl",
    "circuits.jl",
    "distances.jl",
    "fulltomography.jl",
    "gates.jl",
    "getsamples.jl",
    "io.jl",
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
