using Test

@testset "PastaQ.jl" begin
  @testset "$filename" for filename in (
    "utils.jl",
    "optimizers.jl",
    "quantumgates.jl",
    "circuits.jl",
    "circuitops.jl",
    "quantumcircuit.jl",
    "datagen.jl",
    "quantumtomography.jl",
    #"benchmark_qiskit.jl"
  )
    println("Running $filename")
    include(filename)
  end
end
