using Test

@testset "PastaQ.jl" begin
  @testset "$filename" for filename in (
    "utils.jl",
    "optimizers.jl",
    "gates.jl",
    "circuits.jl",
    "circuitops.jl",
    "runcircuit.jl",
    "datagen.jl",
    "tomography.jl",
    "examples.jl", 
    #"benchmark_qiskit.jl"
  )
    println("Running $filename")
    include(filename)
  end
end
