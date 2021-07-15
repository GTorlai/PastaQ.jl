using PastaQ
using PastaQ.ITensors
using Test
using LinearAlgebra
using Random



@testset "noise models: pauli channel" begin
  E = gate("pauli_channel") 
  K = [E[:,:,k] for k in 1:size(E,3)]
  @test size(E,3) == 4
  @test sum([E[:,:,k]' * E[:,:,k] for k in 1:size(E,3)]) ≈ Matrix{Float64}(I,2,2)
  for N in 2:5
    probs = rand(4^N)
    probs = probs ./ sum(probs)
    E = gate("pauli_channel", N; error_probabilities = probs)
    @test size(E,3) == 4^N
    @test sum([E[:,:,k]' * E[:,:,k] for k in 1:size(E,3)]) ≈ Matrix{Float64}(I,1<<N,1<<N)
  end
end

@testset "noise models: bitflip" begin
  E = gate("bit_flip"; p = 0.1) 
  K = [E[:,:,k] for k in 1:size(E,3)]
  @test size(E,3) == 2
  @test sum([E[:,:,k]' * E[:,:,k] for k in 1:size(E,3)]) ≈ Matrix{Float64}(I,2,2)
  for N in 2:5
    E = gate("bit_flip", N; p = 0.1)
    @test size(E,3) == 2^N
    @test sum([E[:,:,k]' * E[:,:,k] for k in 1:size(E,3)]) ≈ Matrix{Float64}(I,1<<N,1<<N)
  end
end

@testset "noise models: phaseflip" begin
  E = gate("phase_flip"; p = 0.1) 
  K = [E[:,:,k] for k in 1:size(E,3)]
  @test size(E,3) == 2
  @test sum([E[:,:,k]' * E[:,:,k] for k in 1:size(E,3)]) ≈ Matrix{Float64}(I,2,2)
  for N in 2:5
    E = gate("phase_flip", N; p = 0.1)
    @test size(E,3) == 2^N
    @test sum([E[:,:,k]' * E[:,:,k] for k in 1:size(E,3)]) ≈ Matrix{Float64}(I,1<<N,1<<N)
  end
end

@testset "noise models: amplitude damping" begin
  E = gate("AD"; γ = 0.1) 
  K = [E[:,:,k] for k in 1:size(E,3)]
  @test size(E,3) == 2
  @test sum([E[:,:,k]' * E[:,:,k] for k in 1:size(E,3)]) ≈ Matrix{Float64}(I,2,2)
  for N in 2:5
    E = gate("AD", N; γ = 0.1)
    @test size(E,3) == 2^N
    @test sum([E[:,:,k]' * E[:,:,k] for k in 1:size(E,3)]) ≈ Matrix{Float64}(I,1<<N,1<<N)
  end

  N = 2
  circuit1 = randomcircuit(N,3; layered = false)
  circuit2 = copy(circuit1)
  push!(circuit1, ("AD",1,(γ=0.1,)))
  push!(circuit1, ("AD",2,(γ=0.1,)))
  push!(circuit2, ("AD",(1,2),(γ=0.1,)))
  ρ₀ = MPO(productstate(N))
  ρ1 = runcircuit(ρ₀, circuit1) 
  ρ2 = runcircuit(ρ₀, circuit2)
  @test PastaQ.array(ρ1) ≈ PastaQ.array(ρ2)
end


@testset "noise models: phase damping" begin
  E = gate("PD"; γ = 0.1) 
  K = [E[:,:,k] for k in 1:size(E,3)]
  @test size(E,3) == 2
  @test sum([E[:,:,k]' * E[:,:,k] for k in 1:size(E,3)]) ≈ Matrix{Float64}(I,2,2)
  for N in 2:5
    E = gate("PD", N; γ = 0.1)
    @test size(E,3) == 2^N
    @test sum([E[:,:,k]' * E[:,:,k] for k in 1:size(E,3)]) ≈ Matrix{Float64}(I,1<<N,1<<N)
  end
  
  N = 2
  circuit1 = randomcircuit(N,3; layered = false)
  circuit2 = copy(circuit1)
  push!(circuit1, ("PD",1,(γ=0.1,)))
  push!(circuit1, ("PD",2,(γ=0.1,)))
  push!(circuit2, ("PD",(1,2),(γ=0.1,)))
  ρ₀ = MPO(productstate(N))
  ρ1 = runcircuit(ρ₀, circuit1) 
  ρ2 = runcircuit(ρ₀, circuit2)
  @test PastaQ.array(ρ1) ≈ PastaQ.array(ρ2)
end


@testset "noise models: depolarizing" begin
  E = gate("DEP"; p = 0.1) 
  K = [E[:,:,k] for k in 1:size(E,3)]
  @test size(E,3) == 4
  @test sum([E[:,:,k]' * E[:,:,k] for k in 1:size(E,3)]) ≈ Matrix{Float64}(I,2,2)
  for N in 2:5
    E = gate("DEP", N; p = 0.1)
    @test size(E,3) == 4^N
    @test sum([E[:,:,k]' * E[:,:,k] for k in 1:size(E,3)]) ≈ Matrix{Float64}(I,1<<N,1<<N)
  end
end


