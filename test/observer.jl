using PastaQ
using ITensors
using Random
using HDF5
using Test

@testset "observer output" begin
  Random.seed!(1234)
  data,Ψ = readsamples("../examples/data/qst_circuit_test.h5")
  N = length(Ψ)     # Number of qubits
  χ = maxlinkdim(Ψ) # Bond dimension of variational MPS
  ψ0 = randomstate(Ψ; χ = χ, σ = 0.1)
  opt = SGD(η = 0.01)
  ψ, obs = tomography(data, ψ0;
                      optimizer = opt,
                      batchsize = 10,
                      epochs = 3,
                      target = Ψ,
                      record = true);
  
  @test length(obs.fidelity) == 3
  @test length(obs.fidelity_bound) == 0
  @test length(obs.frobenius_distance) == 0
  @test length(obs.negative_loglikelihood) == 3
  
  data, ϱ = readsamples("../examples/data/qst_circuit_noisy_test.h5")
  N = length(ϱ)     # Number of qubits
  χ = maxlinkdim(ϱ) # Bond dimension of variational LPDO
  ξ = 2             # Kraus dimension of variational LPDO
  ρ0 = randomstate(ϱ; mixed = true, χ = χ, ξ = ξ, σ = 0.1)
  opt = SGD(η = 0.01)
  ρ, obs = tomography(data,ρ0;
                      optimizer = opt,
                      batchsize = 10,
                      epochs = 3,
                      target = ϱ,
                      record = true);
  @test length(obs.fidelity) == 3
  @test length(obs.fidelity_bound) == 3
  @test length(obs.frobenius_distance) == 3
  @test length(obs.negative_loglikelihood) == 3
  
  data, U = readsamples("../examples/data/qpt_circuit_test.h5")
  N = length(U)     # Number of qubits
  χ = maxlinkdim(U) # Bond dimension of variational MPS
  opt = SGD(η = 0.1)
  V0 = randomprocess(U; mixed = false, χ = χ)
  V, obs = tomography(data, V0;
                      optimizer = opt,
                      batchsize = 10,
                      epochs = 3,
                      target = U,
                      record = true)

  @test length(obs.fidelity) == 3
  @test length(obs.fidelity_bound) == 0
  @test length(obs.frobenius_distance) == 0
  @test length(obs.negative_loglikelihood) == 3
  
  # Noisy circuit
  Random.seed!(1234)
  data, ϱ = readsamples("../examples/data/qpt_circuit_noisy_test.h5")
  N = length(ϱ)
  χ = 8
  ξ = 2
  Λ0 = randomprocess(ϱ; mixed = true, χ = χ, ξ = ξ, σ = 0.1)
  opt = SGD(η = 0.1)
  Λ, obs = tomography(data, Λ0;
                      optimizer = opt,
                      mixed = true,
                      batchsize = 10,
                      epochs = 3,
                      target = ϱ,
                      record = true);
  @test length(obs.fidelity) == 3
  @test length(obs.fidelity_bound) == 3
  @test length(obs.frobenius_distance) == 3
  @test length(obs.negative_loglikelihood) == 3
end
