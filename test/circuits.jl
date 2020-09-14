using PastaQ
using ITensors
using Test
using Random

@testset "circuits: appendlayer!" begin
  N = 5
  gates = Tuple[]
  appendlayer!(gates, "H", N)
  @test length(gates) == N
  for j in 1:N
    @test gates[j] isa Tuple{String,Int64}
    @test gates[j][1] == "H"
    @test gates[j][2] == j
  end
end

@testset "circuits: randU layer" begin
  N = 5
  gates = Tuple[]
  appendlayer!(gates, "randU", N)
  @test length(gates) == N
  for j in 1:N
    @test typeof(gates[j]) == Tuple{String,Int64}
    @test gates[j][1] == "randU"
    @test gates[j][2] == j
  end
end

