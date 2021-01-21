using PastaQ
using PastaQ.ITensors
using Test
using LinearAlgebra

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


@testset "qubits initialization" begin
  N = 1
  ψ = qubits(N)
  @test length(ψ) == 1
  @test typeof(ψ) == MPS
  @test length(inds(ψ[1],"Link")) == 0
  @test PastaQ.fullvector(ψ) ≈ [1, 0]
  N = 5
  ψ = qubits(N)
  @test length(ψ) == 5
  ψ_vec = PastaQ.fullvector(ψ)
  exact_vec = zeros(1<<N)
  exact_vec[1] = 1.0
  @test ψ_vec ≈ exact_vec
end

@testset "circuit MPO initialization" begin
  N = 5
  U = identity_mpo(N)
  @test length(U) == N
  U_mat = PastaQ.fullmatrix(U)
  exact_mat = Matrix{ComplexF64}(I, 1<<N, 1<<N)
  @test U_mat ≈ exact_mat
end

@testset "Density matrix initialization" begin
  N = 5
  ρ1 = qubits(N,mixed=true)
  @test length(ρ1) == N
  @test typeof(ρ1) == MPO
  ψ = qubits(N)
  ρ2 = qubits(N,mixed=true)
  @test PastaQ.fullmatrix(ρ1) ≈ PastaQ.fullmatrix(ρ2)
  exact_mat = zeros(1<<N,1<<N)
  exact_mat[1,1] = 1.0
  @test PastaQ.fullmatrix(ρ2) ≈ exact_mat
end

@testset "reset qubits" begin
  N = 5
  depth = 5
  gates = randomcircuit(N,depth)
  ψ0 = qubits(N)
  ψ = runcircuit(ψ0,gates)
  
  resetqubits!(ψ)
  psi_vec = PastaQ.fullvector(ψ)

  exact_vec = zeros(1<<N)
  exact_vec[1] = 1.0
  @test psi_vec ≈ exact_vec

  
  ρ0 = qubits(N,mixed=true)
  ρ = runcircuit(ρ0,gates)
  
  resetqubits!(ρ)
  ρ_mat = PastaQ.fullmatrix(ρ)

  exact_mat = zeros(1<<N,1<<N)
  exact_mat[1,1] = 1.0
  @test exact_mat ≈ ρ_mat
end


@testset "runcircuit: unitary quantum circuit" begin
  N = 3
  depth = 4
  gates = randomcircuit(N,depth)
  ngates = N*depth + depth÷2 * (N-1)
  @test length(gates) == ngates
  
  #Pure state, noiseless circuit
  ψ0 = qubits(N)
  ψ = runcircuit(ψ0,gates)
  @test prod(ψ) ≈ runcircuit(prod(ψ0),buildcircuit(ψ0,gates))
  @test array(prod(ψ)) ≈ array(prod(runcircuit(N,gates)))
  @test array(prod(ψ)) ≈ array(prod(runcircuit(gates)))
  
  # Mixed state, noiseless circuit
  ρ0 = qubits(N,mixed=true) 
  ρ = runcircuit(ρ0,gates)
  @test prod(ρ) ≈ runcircuit(prod(ρ0),buildcircuit(ρ0,gates); apply_dag=true)
  
end

@testset "runcircuit: noisy quantum circuit" begin
  N = 5
  depth = 4
  gates = randomcircuit(N,depth)
  ngates = N*depth + depth÷2 * (N-1)
  @test length(gates) == ngates

  # Pure state, noisy circuit
  ψ0 = qubits(N)
  ρ = runcircuit(ψ0, gates; noise = ("depolarizing", (p = 0.1,)))
  ρ0 = MPO(ψ0)
  U = buildcircuit(ρ0, gates; noise = ("depolarizing", (p = 0.1,)))
  @disable_warn_order begin
    @test prod(ρ) ≈ runcircuit(prod(ρ0), U; apply_dag=true)
    
    ## Mixed state, noisy circuit
    ρ0 = qubits(N, mixed = true)
    ρ = runcircuit(ρ0, gates; noise = ("depolarizing", (p = 0.1,)))
    U = buildcircuit(ρ0, gates, noise = ("depolarizing", (p = 0.1,)))
    @test prod(ρ) ≈ runcircuit(prod(ρ0), U; apply_dag=true)
  end

end

@testset "runcircuit: inverted gate order" begin
  N = 8
  gates = randomcircuit(N,2)
  
  for n in 1:10
    s1 = rand(2:N)
    s2 = s1-1
    push!(gates,("CX", (s1,s2)))
  end
  ψ0 = qubits(N)
  ψ = runcircuit(ψ0, gates)
  @test prod(ψ) ≈ runcircuit(prod(ψ0),buildcircuit(ψ0,gates))

end

@testset "runcircuit: long range gates" begin
  N = 8
  gates = randomcircuit(N,2)
  
  for n in 1:10
    s1 = rand(1:N)
    s2 = rand(1:N)
    while s2 == s1
      s2 = rand(1:N)
    end
    push!(gates,("CX", (s1,s2)))
  end
  ψ0 = qubits(N)
  ψ = runcircuit(ψ0,gates)
  @test prod(ψ) ≈ runcircuit(prod(ψ0),buildcircuit(ψ0,gates)) 
  
end


