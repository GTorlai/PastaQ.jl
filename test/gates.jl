using PastaQ
using ITensors
using Test
using LinearAlgebra

@testset "Gate generation" begin
  i = Index(2)
  j = Index(2)

  g = gate("I", i)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  g = gate("X", i)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  g = gate("Y", i)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  g = gate("Z", i)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  g = gate("H", i)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  g = gate("S", i)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  g = gate("T", i)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  g = state("X+", i)
  @test plev(ind(g, 1)) == 0

  g = state("X-", i)
  @test plev(ind(g, 1)) == 0

  g = state("Y+", i)
  @test plev(ind(g, 1)) == 0

  g = state("Y-", i)
  @test plev(ind(g, 1)) == 0

  g = state("Z+", i)
  @test plev(ind(g, 1)) == 0

  g = state("Z-", i)
  @test plev(ind(g, 1)) == 0

  θ = π * rand()
  g = gate("Rx", i; θ=θ)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  θ = π * rand()
  g = gate("Ry", i; θ=θ)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  ϕ = 2π * rand()
  g = gate("Rz", i; ϕ=ϕ)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  angles = rand(3)
  θ = π * angles[1]
  ϕ = 2π * angles[2]
  λ = 2π * angles[3]
  g = gate("Rn", i; θ=θ, ϕ=ϕ, λ=λ)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  g = gate("SWAP", i, j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4), (2, 2, 2, 2))

  g = gate("CX", i, j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4), (2, 2, 2, 2))

  g = gate("CY", i, j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4), (2, 2, 2, 2))

  g = gate("CZ", i, j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4), (2, 2, 2, 2))

  g = gate("randU", i)
  @test hasinds(g, i', i)
  Id = g * swapprime(dag(g)', 2 => 1)
  for n in 1:dim(i), n′ in 1:dim(i)
    if n == n′
      @test Id[n, n′] ≈ 1
    else
      @test Id[n, n′] ≈ 0 atol = 1e-15
    end
  end

  g = gate("randU", i, j)
  @test hasinds(g, i', j', i, j)
  Id = g * swapprime(dag(g)', 2 => 1)
  for n in 1:dim(i), m in 1:dim(j), n′ in 1:dim(i), m′ in 1:dim(j)
    if (n, m) == (n′, m′)
      @test Id[n, m, n′, m′] ≈ 1
    else
      @test Id[n, m, n′, m′] ≈ 0 atol = 1e-14
    end
  end

  @testset "Custom gate with long name" begin
    PastaQ.gate(::GateName"my_favorite_gate") = [0.11 0.12; 0.21 0.22]
    s = Index(2, "Qubit, Site")
    gate("my_favorite_gate", s)
    g = gate("my_favorite_gate", s)
    @test g[s' => 1, s => 1] == 0.11
    @test g[s' => 1, s => 2] == 0.12
    @test g[s' => 2, s => 1] == 0.21
    @test g[s' => 2, s => 2] == 0.22
  end
end
