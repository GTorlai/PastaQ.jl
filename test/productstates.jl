using ITensors
using PastaQ
using Test

@testset "productstate" begin
  s = siteinds("Qubit", 4)

  states0 = [state("0", s[n]) for n in 1:length(s)]
  states1 = [state("1", s[n]) for n in 1:length(s)]
  statesXp = [state("X+", s[n]) for n in 1:length(s)]
  statesYm = [state("Y-", s[n]) for n in 1:length(s)]

  ψ0 = productstate(s)
  @test ψ0 isa MPS
  @test eltype(ψ0) == ITensor
  @test PastaQ.promote_leaf_eltypes(ψ0) == Float64
  @test ψ0 ≈ prod(states0)

  @test productstate(s, 0) ≈ ψ0
  @test productstate(s, "0") ≈ ψ0

  ψXp = productstate(s, "X+")
  @test ψXp isa MPS
  @test PastaQ.promote_leaf_eltypes(ψXp) == Float64
  @test ψXp ≈ prod(statesXp)

  ψYm = productstate(s, "Y-")
  @test ψYm isa MPS
  @test PastaQ.promote_leaf_eltypes(ψYm) == ComplexF64
  @test prod(ψYm) ≈ prod(statesYm)

  @test productstate(s, fill("0", length(s))) ≈ ψ0
  @test productstate(s, fill(0, length(s))) ≈ ψ0
  @test productstate(s, fill("X+", length(s))) ≈ prod(statesXp)
  @test productstate(s, fill("Y-", length(s))) ≈ prod(statesYm)

  @test productstate(s, n -> isodd(n) ? 1 : 0) ≈ states1[1] * states0[2] * states1[3] * states0[4]
  @test productstate(s, n -> isodd(n) ? "1" : "0") ≈ states1[1] * states0[2] * states1[3] * states0[4]
  @test productstate(s, n -> isodd(n) ? "Y-" : "X+") ≈ statesYm[1] * statesXp[2] * statesYm[3] * statesXp[4]
end

@testset "productoperator" begin
  s = siteinds("Qubit", 4)

  gatesI = [gate("Id", s, n) for n in 1:length(s)]

  I = productoperator(s)
  @test PastaQ.promote_leaf_eltypes(I) == Float64
  @test prod(I) ≈ prod(gatesI)
  @test I ≈ prod(gatesI)
end

