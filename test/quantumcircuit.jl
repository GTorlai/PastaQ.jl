include("../src/PastaQ.jl")
using Main.PastaQ
using HDF5, JLD
using ITensors
using Test
using LinearAlgebra


@testset "qubits initialization" begin
  N = 1
  psi =InitializeQubits(N)
  @test length(psi) == 1
  @test length(inds(psi[1],"Link")) == 0
  N = 5
  psi =InitializeQubits(N)
  @test length(psi) == 5
  psi_vec = FullVector(psi)
  exact_vec = zeros(1<<N)
  exact_vec[1] = 1.0
  exact_vec = itensor(exact_vec,inds(psi_vec))
  @test psi_vec ≈ exact_vec
end

@testset "circuit initialization" begin
  N=5
  U = InitializeCircuit(N)
  @test length(U) == 5
  identity = itensor(reshape([1 0;0 1],(1,2,2)),inds(U[1]))
  @test U[1] ≈ identity
  for s in 2:N-1
    identity = itensor(reshape([1 0;0 1],(1,1,2,2)),inds(U[s]))
    @test U[s] ≈ identity
  end
  identity = itensor(reshape([1 0;0 1],(1,2,2)),inds(U[N]))
  @test U[N] ≈ identity
end

