using ITensors
using PastaQ
using Test

@testset "depracated" begin
  N = 5

  sites = siteinds("Qubit", N)

  q = qubits(N)
  @test q isa MPS
  @test length(q) == N

  q = qubits(sites)
  @test q isa MPS
  @test length(q) == N
  @test q ≈ productstate(sites)

  ψ = randomstate(sites)
  q = qubits(ψ)
  @test q isa MPS
  @test length(q) == N
  @test siteinds(ψ) == siteinds(q)

  q = qubits(sites, ["Z+", "Z-", "X+", "Y-", "X+"])
  @test q isa MPS
  @test length(q) == N
  @test q ≈ productstate(sites, ["Z+", "Z-", "X+", "Y-", "X+"])
end
