using Test

@testset "PastaQ.jl" begin
  @testset "$filename" for filename in (
    "utils.jl",
    "gates.jl",
    "circuits.jl",
    "circuitops.jl",
    "runcircuit.jl",
    "datagen.jl",
    "optimizers.jl",
    "distances.jl",
    "tomography.jl",
    "examples.jl", 
    #"benchmark_qiskit.jl"
  )
    println("Running $filename")
    include(filename)
  end
end
