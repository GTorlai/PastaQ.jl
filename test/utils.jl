using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random

@testset "hilbertspace" begin
  N = 5
  ψ = productstate(N)
  ρ = MPO(productstate(ψ))
  Λ = randomstate(ψ; mixed=true)

  @test PastaQ.hilbertspace(ψ) == siteinds(ψ)
  @test PastaQ.hilbertspace(ψ) == PastaQ.hilbertspace(ρ)
  @test PastaQ.hilbertspace(ψ) == PastaQ.hilbertspace(Λ)
end

@testset "choi tags and MPO/MPS conversion" begin
  N = 4
  circuit = randomcircuit(4; depth=4)

  U = runcircuit(circuit; process=true)
  ρ = PastaQ.choimatrix(PastaQ.hilbertspace(U), circuit; noise=("DEP", (p=0.01,)))
  Λ = randomprocess(4; mixed=true)

  @test PastaQ.ischoi(ρ) == true
  @test PastaQ.ischoi(U) == false
  @test PastaQ.ischoi(Λ) == true
  @test PastaQ.haschoitags(U) == false
  @test PastaQ.haschoitags(ρ) == true
  @test PastaQ.haschoitags(Λ) == true

  Ψ = PastaQ.choitags(U)
  @test hastags(inds(Ψ[1]), "Input") == true
  @test hastags(inds(Ψ[1]), "Output") == true
  @test plev(firstind(Ψ[1]; tags="Input")) == 0
  @test plev(firstind(Ψ[1]; tags="Output")) == 0

  V = PastaQ.mpotags(Ψ)
  @test hastags(inds(V[1]), "Input") == false
  @test hastags(inds(V[1]), "Output") == false
  @test plev(inds(V[1]; tags="Qubit")[1]) == 1
  @test plev(inds(V[1]; tags="Qubit")[2]) == 0

  Ψ = PastaQ.unitary_mpo_to_choi_mps(U)
  @test Ψ isa MPS
  @test hastags(inds(Ψ[1]), "Input") == true
  @test hastags(inds(Ψ[1]), "Output") == true
  @test plev(firstind(Ψ[1]; tags="Input")) == 0
  @test plev(firstind(Ψ[1]; tags="Output")) == 0

  V = PastaQ.choi_mps_to_unitary_mpo(Ψ)
  @test V isa MPO
  @test hastags(inds(V[1]), "Input") == false
  @test hastags(inds(V[1]), "Output") == false
  @test plev(inds(V[1]; tags="Qubit")[1]) == 1
  @test plev(inds(V[1]; tags="Qubit")[2]) == 0
end
