using PastaQ
using PastaQ.ITensors
using Test
using LinearAlgebra
using Random

function state_to_int(state::Array)
  index = 0
  for j in 1:length(state)
    index += 2^(j - 1) * state[length(state) + 1 - j]
  end
  return index
end

function empiricalprobability(samples::Matrix)
  prob = zeros((1 << size(samples)[2]))
  for n in 1:size(samples)[1]
    sample = samples[n, :]
    index = state_to_int(sample)
    prob[index + 1] += 1
  end
  prob = prob / size(samples)[1]
  return prob
end

@testset "productstate initialization" begin
  N = 1
  ψ = productstate(N)
  @test length(ψ) == 1
  @test typeof(ψ) == MPS
  @test length(inds(ψ[1], "Link")) == 0
  @test PastaQ.array(ψ) ≈ [1, 0]
  N = 5
  ψ = productstate(N)
  @test length(ψ) == 5
  ψ_vec = PastaQ.array(ψ)
  exact_vec = zeros(1 << N)
  exact_vec[1] = 1.0
  @test ψ_vec ≈ exact_vec
end

@testset "circuit MPO initialization" begin
  N = 5
  U = productoperator(N)
  @test length(U) == N
  U_mat = PastaQ.array(U)
  exact_mat = Matrix{ComplexF64}(I, 1 << N, 1 << N)
  @test U_mat ≈ exact_mat
end

@testset "Density matrix initialization" begin
  N = 5
  ρ1 = MPO(productstate(N))
  @test length(ρ1) == N
  @test typeof(ρ1) == MPO
  ψ = productstate(N)
  ρ2 = MPO(productstate(N))
  @test PastaQ.array(ρ1) ≈ PastaQ.array(ρ2)
  exact_mat = zeros(1 << N, 1 << N)
  exact_mat[1, 1] = 1.0
  @test PastaQ.array(ρ2) ≈ exact_mat
end

#@testset "reset productstate" begin
#  N = 5
#  depth = 5
#  gates = randomcircuit(N,depth)
#  ψ0 = productstate(N)
#  ψ = runcircuit(ψ0,gates)
#  
#  resetproductstate!(ψ)
#  psi_vec = PastaQ.array(ψ)
#
#  exact_vec = zeros(1<<N)
#  exact_vec[1] = 1.0
#  @test psi_vec ≈ exact_vec
#
#  
#  ρ0 = MPO(productstate(N))
#  ρ = runcircuit(ρ0,gates)
#  
#  resetproductstate!(ρ)
#  ρ_mat = PastaQ.array(ρ)
#
#  exact_mat = zeros(1<<N,1<<N)
#  exact_mat[1,1] = 1.0
#  @test exact_mat ≈ ρ_mat
#end

@testset "runcircuit: unitary quantum circuit" begin
  N = 3
  depth = 4
  gates = randomcircuit(N, depth; layered=false)
  #Pure state, noiseless circuit
  ψ0 = productstate(N)
  ψ = runcircuit(ψ0, gates)
  @test prod(ψ) ≈ runcircuit(prod(ψ0), buildcircuit(ψ0, gates))
  @test PastaQ.array(prod(ψ)) ≈ PastaQ.array(prod(runcircuit(N, gates)))
  @test PastaQ.array(prod(ψ)) ≈ PastaQ.array(prod(runcircuit(gates)))

  # Mixed state, noiseless circuit
  ρ0 = MPO(productstate(N))
  ρ = runcircuit(ρ0, gates)
  @test prod(ρ) ≈ runcircuit(prod(ρ0), buildcircuit(ρ0, gates); apply_dag=true)
end

@testset "runcircuit: noisy quantum circuit" begin
  N = 5
  depth = 4
  gates = randomcircuit(N, depth; layered=false)

  # Pure state, noisy circuit
  ψ0 = productstate(N)
  ρ = runcircuit(ψ0, gates; noise=("depolarizing", (p=0.1,)))
  ρ0 = MPO(ψ0)
  U = buildcircuit(ρ0, gates; noise=("depolarizing", (p=0.1,)))
  @disable_warn_order begin
    @test prod(ρ) ≈ runcircuit(prod(ρ0), U; apply_dag=true)

    ## Mixed state, noisy circuit
    ρ0 = MPO(productstate(N))
    ρ = runcircuit(ρ0, gates; noise=("depolarizing", (p=0.1,)))
    U = buildcircuit(ρ0, gates; noise=("depolarizing", (p=0.1,)))
    @test prod(ρ) ≈ runcircuit(prod(ρ0), U; apply_dag=true)
  end
end

@testset "alternative noise definition" begin
  N = 5
  depth = 4
  circuit0 = randomcircuit(N, depth; twoqubitgates="CX", onequbitgates="Rn", layered=false)
  ρ0 = runcircuit(circuit0; noise=("DEP", (p=0.01,)))

  ψ = productstate(ρ0)
  circuit = []
  for g in circuit0
    push!(circuit, g)
    ns = g[2]
    for n in ns
      push!(circuit, ("DEP", n, (p=0.01,)))
    end
  end
  ρ = runcircuit(ψ, circuit)
  @test PastaQ.array(ρ0) ≈ PastaQ.array(ρ)
end

@testset "runcircuit: inverted gate order" begin
  N = 8
  gates = randomcircuit(N, 2; layered=false)

  for n in 1:10
    s1 = rand(2:N)
    s2 = s1 - 1
    push!(gates, ("CX", (s1, s2)))
  end
  ψ0 = productstate(N)
  ψ = runcircuit(ψ0, gates)
  @test prod(ψ) ≈ runcircuit(prod(ψ0), buildcircuit(ψ0, gates))
end

@testset "runcircuit: long range gates" begin
  N = 8
  gates = randomcircuit(N, 2; layered=false)

  for n in 1:10
    s1 = rand(1:N)
    s2 = rand(1:N)
    while s2 == s1
      s2 = rand(1:N)
    end
    push!(gates, ("CX", (s1, s2)))
  end
  ψ0 = productstate(N)
  ψ = runcircuit(ψ0, gates)
  @test prod(ψ) ≈ runcircuit(prod(ψ0), buildcircuit(ψ0, gates))
end

@testset "layered circuit" begin
  N = 4
  depth = 10
  ψ0 = productstate(N)

  Random.seed!(1234)
  circuit = randomcircuit(N, depth)
  ψ = runcircuit(ψ0, circuit)
  Random.seed!(1234)
  circuit = randomcircuit(N, depth)
  @test prod(ψ) ≈ prod(runcircuit(ψ0, circuit))

  Random.seed!(1234)
  circuit = randomcircuit(N, depth)
  ρ = runcircuit(ψ0, circuit; noise=("depolarizing", (p=0.1,)))
  Random.seed!(1234)
  circuit = randomcircuit(N, depth)
  @test prod(ρ) ≈ prod(runcircuit(ψ0, circuit; noise=("depolarizing", (p=0.1,))))
end
