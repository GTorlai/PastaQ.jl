using PastaQ
using PastaQ.ITensors
using Test
using LinearAlgebra
using Random

@testset "noise models: pauli channel" begin
  E = gate("pauli_channel")
  K = [E[:, :, k] for k in 1:size(E, 3)]
  @test size(E, 3) == 4
  @test sum([E[:, :, k]' * E[:, :, k] for k in 1:size(E, 3)]) ≈ Matrix{Float64}(I, 2, 2)
  for N in 2:5
    probs = rand(4^N)
    probs = probs ./ sum(probs)
    E = gate("pauli_channel", Tuple(repeat([2], N)); error_probabilities=probs)
    @test size(E, 3) == 4^N
    @test sum([E[:, :, k]' * E[:, :, k] for k in 1:size(E, 3)]) ≈ Matrix{Float64}(I, 1 << N, 1 << N)
  end
end

@testset "noise models: bitflip" begin
  E = gate("bit_flip"; p=0.1)
  K = [E[:, :, k] for k in 1:size(E, 3)]
  @test size(E, 3) == 2
  @test sum([E[:, :, k]' * E[:, :, k] for k in 1:size(E, 3)]) ≈ Matrix{Float64}(I, 2, 2)
  for N in 2:5
    E = gate("bit_flip", Tuple(repeat([2], N)); p=0.1)
    @test size(E, 3) == 2^N
    @test sum([E[:, :, k]' * E[:, :, k] for k in 1:size(E, 3)]) ≈ Matrix{Float64}(I, 1 << N, 1 << N)
  end
end

@testset "noise models: phaseflip" begin
  E = gate("phase_flip"; p=0.1)
  K = [E[:, :, k] for k in 1:size(E, 3)]
  @test size(E, 3) == 2
  @test sum([E[:, :, k]' * E[:, :, k] for k in 1:size(E, 3)]) ≈ Matrix{Float64}(I, 2, 2)
  for N in 2:5
    E = gate("phase_flip", Tuple(repeat([2], N)); p=0.1)
    @test size(E, 3) == 2^N
    @test sum([E[:, :, k]' * E[:, :, k] for k in 1:size(E, 3)]) ≈ Matrix{Float64}(I, 1 << N, 1 << N)
  end
end

@testset "noise models: amplitude damping" begin
  E = gate("AD"; γ=0.1)
  K = [E[:, :, k] for k in 1:size(E, 3)]
  @test size(E, 3) == 2
  @test sum([E[:, :, k]' * E[:, :, k] for k in 1:size(E, 3)]) ≈ Matrix{Float64}(I, 2, 2)
  for N in 2:5
    E = gate("AD", Tuple(repeat([2], N)); γ=0.1)
    @test size(E, 3) == 2^N
    @test sum([E[:, :, k]' * E[:, :, k] for k in 1:size(E, 3)]) ≈ Matrix{Float64}(I, 1 << N, 1 << N)
  end

  N = 2
  circuit1 = randomcircuit(N; depth=3, layered=false)
  circuit2 = copy(circuit1)
  push!(circuit1, ("AD", 1, (γ=0.1,)))
  push!(circuit1, ("AD", 2, (γ=0.1,)))
  push!(circuit2, ("AD", (1, 2), (γ=0.1,)))
  ρ₀ = MPO(productstate(N))
  ρ1 = runcircuit(ρ₀, circuit1)
  ρ2 = runcircuit(ρ₀, circuit2)
  @test PastaQ.array(ρ1) ≈ PastaQ.array(ρ2)
end

@testset "noise models: phase damping" begin
  E = gate("PD"; γ=0.1)
  K = [E[:, :, k] for k in 1:size(E, 3)]
  @test size(E, 3) == 2
  @test sum([E[:, :, k]' * E[:, :, k] for k in 1:size(E, 3)]) ≈ Matrix{Float64}(I, 2, 2)
  for N in 2:5
    E = gate("PD", Tuple(repeat([2], N)); γ=0.1)
    @test size(E, 3) == 2^N
    @test sum([E[:, :, k]' * E[:, :, k] for k in 1:size(E, 3)]) ≈ Matrix{Float64}(I, 1 << N, 1 << N)
  end

  N = 2
  circuit1 = randomcircuit(N; depth=3, layered=false)
  circuit2 = copy(circuit1)
  push!(circuit1, ("PD", 1, (γ=0.1,)))
  push!(circuit1, ("PD", 2, (γ=0.1,)))
  push!(circuit2, ("PD", (1, 2), (γ=0.1,)))
  ρ₀ = MPO(productstate(N))
  ρ1 = runcircuit(ρ₀, circuit1)
  ρ2 = runcircuit(ρ₀, circuit2)
  @test PastaQ.array(ρ1) ≈ PastaQ.array(ρ2)
end

@testset "noise models: depolarizing" begin
  E = gate("DEP"; p=0.1)
  K = [E[:, :, k] for k in 1:size(E, 3)]
  @test size(E, 3) == 4
  @test sum([E[:, :, k]' * E[:, :, k] for k in 1:size(E, 3)]) ≈ Matrix{Float64}(I, 2, 2)
  for N in 2:5
    E = gate("DEP", Tuple(repeat([2], N)); p=0.1)
    @test size(E, 3) == 4^N
    @test sum([E[:, :, k]' * E[:, :, k] for k in 1:size(E, 3)]) ≈ Matrix{Float64}(I, 1 << N, 1 << N)
  end
end

@testset "insertnoise: iid noise" begin
  N = 6
  depth = 4

  circuit = randomcircuit(
    N; depth=depth, twoqubitgates=["CX", "CZ"], onequbitgates=["Rn", "X"], layered=false
  )
  ngates = length(circuit)
  nCZ, nCX, nR, nX = 0, 0, 0, 0
  for g in circuit
    g[1] == "CX" && (nCX += 1)
    g[1] == "CZ" && (nCZ += 1)
    g[1] == "Rn" && (nR += 1)
    g[1] == "X" && (nX += 1)
  end

  noisemodel = ("DEP", (p=0.01,))
  noisycircuit = insertnoise(circuit, noisemodel)
  @test length(noisycircuit) == 2 * ngates
  for k in 2:2:length(circuit)
    g = noisycircuit[k]
    @test g[1] == noisemodel[1]
    @test g[3] == noisemodel[2]
  end

  noisycircuit = insertnoise(circuit, noisemodel; gate="CX")
  @test length(noisycircuit) == ngates + nCX
  for k in 1:length(circuit)
    g = noisycircuit[k]
    if g[1] == "CX"
      @test noisycircuit[k + 1][1] == noisemodel[1]
      @test noisycircuit[k + 1][3] == noisemodel[2]
    else
      @test noisycircuit[k + 1][1] != noisemodel[1]
    end
  end
  noisycircuit = insertnoise(circuit, noisemodel; gate=["CX", "CZ"])
  @test length(noisycircuit) == ngates + nCX + nCZ
end

@testset "insertnoise: one and two qubit noise" begin
  N = 6
  depth = 4

  circuit = randomcircuit(
    N; depth=depth, twoqubitgates=["CX", "CZ"], onequbitgates=["Rn", "X"], layered=false
  )
  ngates = length(circuit)
  nCZ, nCX, nR, nX = 0, 0, 0, 0
  for g in circuit
    g[1] == "CX" && (nCX += 1)
    g[1] == "CZ" && (nCZ += 1)
    g[1] == "Rn" && (nR += 1)
    g[1] == "X" && (nX += 1)
  end

  noise1Q = ("AD", (γ=0.01,))
  noise2Q = ("DEP", (p=0.05,))

  noisemodel = (1 => noise1Q, 2 => noise2Q)
  noisycircuit = insertnoise(circuit, noisemodel)
  @test length(noisycircuit) == 2 * ngates
  for k in 1:2:length(circuit)
    g = noisycircuit[k]
    if g[2] isa Int
      @test noisycircuit[k + 1][1] == noise1Q[1]
      @test noisycircuit[k + 1][3] == noise1Q[2]
    else
      @test noisycircuit[k + 1][1] == noise2Q[1]
      @test noisycircuit[k + 1][3] == noise2Q[2]
    end
  end
end
