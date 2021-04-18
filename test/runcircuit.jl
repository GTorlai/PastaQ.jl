using PastaQ
using PastaQ.ITensors
using Test
using LinearAlgebra
using Random

function state_to_int(state::Array)
  index = 0
  for j in 1:length(state)
    index += 2^(j-1)*state[length(state)+1-j]
  end
  return index
end

function empiricalprobability(samples::Matrix)
  prob = zeros((1<<size(samples)[2]))
  for n in 1:size(samples)[1]
    sample = samples[n,:]
    index = state_to_int(sample)
    prob[index+1] += 1
  end
  prob = prob / size(samples)[1]
  return prob
end


@testset "productstate initialization" begin
  N = 1
  ψ = productstate(N)
  @test length(ψ) == 1
  @test typeof(ψ) == MPS
  @test length(inds(ψ[1],"Link")) == 0
  @test PastaQ.array(ψ) ≈ [1, 0]
  N = 5
  ψ = productstate(N)
  @test length(ψ) == 5
  ψ_vec = PastaQ.array(ψ)
  exact_vec = zeros(1<<N)
  exact_vec[1] = 1.0
  @test ψ_vec ≈ exact_vec
end

@testset "circuit MPO initialization" begin
  N = 5
  U = productoperator(N)
  @test length(U) == N
  U_mat = PastaQ.array(U)
  exact_mat = Matrix{ComplexF64}(I, 1<<N, 1<<N)
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
  exact_mat = zeros(1<<N,1<<N)
  exact_mat[1,1] = 1.0
  @test PastaQ.array(ρ2) ≈ exact_mat
end

@testset "runcircuit: unitary quantum circuit" begin
  N = 3
  depth = 4
  circuit = randomcircuit(N,depth; layered = false)
  #Pure state, noiseless circuit
  ψ₀ = productstate(N)
  ψ₀dense = prod(ψ₀)

  ψdense = runcircuit(ψ₀dense,circuit)
  @test ψdense ≈ runcircuit(siteinds(ψ₀), circuit; exact = true) 
  
  ψ = runcircuit(siteinds(ψ₀),circuit)
  @test prod(ψ) ≈ ψdense
  ψ = runcircuit(ψ₀,circuit)
  @test prod(ψ) ≈ ψdense
  
  # Mixed state, noiseless circuit
  ρ₀ = MPO(productstate(N))
  ρ₀dense = prod(ρ₀)

  ρdense = runcircuit(ρ₀dense, circuit)
  ρ = runcircuit(ρ₀, circuit)
  @test prod(ρ) ≈ ρdense
  
end

@testset "runcircuit: noisy quantum circuit" begin
  N = 3
  depth = 2
  gates = randomcircuit(N,depth; layered = false)

  ψ0 = productstate(N)
  ρ = runcircuit(ψ0, gates; noise = ("depolarizing", (p = 0.1,)))
  
  ρdense = runcircuit(siteinds(ψ0), gates; noise = ("depolarizing", (p = 0.1,)), exact = true)
  @test ρdense ≈ prod(ρ)
  
  ρ = runcircuit(MPO(ψ0), gates;  noise = ("depolarizing", (p = 0.1,)))
  @test ρdense ≈ prod(ρ)
  
  ρdense = runcircuit(prod(MPO(ψ0)), gates;  noise = ("depolarizing", (p = 0.1,)))
  @test ρdense ≈ prod(ρ)
  
end
#
#@testset "alternative noise definition" begin
#  N = 5
#  depth = 4
#  circuit0 = randomcircuit(N,depth; twoqubitgates = "CX", onequbitgates = "Rn", layered = false)
#  ρ0 = runcircuit(circuit0; noise = ("DEP", (p=0.01,)))
#
#  ψ = productstate(ρ0)
#  circuit = []
#  for g in circuit0
#    push!(circuit,g)
#    ns = g[2]
#    push!(circuit,("DEP",ns,(p=0.01,)))
#  end
#  ρ = runcircuit(ψ, circuit)
#  @test PastaQ.array(ρ0) ≈ PastaQ.array(ρ)
#end

@testset "runcircuit: inverted gate order" begin
  N = 8
  gates = randomcircuit(N,2; layered = false)
  
  for n in 1:10
    s1 = rand(2:N)
    s2 = s1-1
    push!(gates,("CX", (s1,s2)))
  end
  ψ0 = productstate(N)
  ψ = runcircuit(ψ0, gates)
  @test prod(ψ) ≈ runcircuit(siteinds(ψ0), gates; exact = true)
end

@testset "runcircuit: long range gates" begin
  N = 8
  gates = randomcircuit(N,2; layered = false)
  
  for n in 1:10
    s1 = rand(1:N)
    s2 = rand(1:N)
    while s2 == s1
      s2 = rand(1:N)
    end
    push!(gates,("CX", (s1,s2)))
  end
  ψ0 = productstate(N)
  ψ = runcircuit(ψ0,gates)
  @test prod(ψ) ≈ runcircuit(siteinds(ψ0), gates; exact = true)
  
end

@testset "layered circuit" begin
  N = 4
  depth = 10
  ψ0 = productstate(N)
  
  Random.seed!(1234)
  circuit = randomcircuit(N, depth)
  ψ = runcircuit(ψ0,circuit)
  Random.seed!(1234)
  circuit = randomcircuit(N, depth)
  @test prod(ψ) ≈ prod(runcircuit(ψ0,circuit))
  
  Random.seed!(1234)
  circuit = randomcircuit(N, depth)
  ρ = runcircuit(ψ0, circuit; noise = ("depolarizing",(p=0.1,)))
  Random.seed!(1234)
  circuit = randomcircuit(N, depth)
  @test prod(ρ) ≈ prod(runcircuit(ψ0, circuit;noise = ("depolarizing",(p=0.1,))))

end

