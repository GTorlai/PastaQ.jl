using Test

@testset "PastaQ.jl" begin
  @testset "$filename" for filename in (
    "quantumgates.jl",
    "quantumcircuit.jl"
  )
    println("Running $filename")
    include(filename)
  end
end

