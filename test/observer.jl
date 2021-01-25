using PastaQ
using ITensors
using Random
using Test

@testset "circuit observer" begin
  N = 10
  depth = 8
  circuit = randomcircuit(N,8; layered = true, seed = 1234)

  function measure_pauli(ψ::MPS, site::Int, pauli::String)
    ψ = orthogonalize!(copy(ψ), site)
    ϕ = ψ[site]
    obs_op = gate(pauli, firstsiteind(ψ, site))
    T = noprime(ϕ * obs_op)
    return real((dag(T) * ϕ)[])
  end
  pauliX2(ψ::MPS) = measure_pauli(ψ, 2, "X")
  pauliYs(ψ::MPS) = [measure_pauli(ψ, n, "Y") for n in 1:length(ψ)]
  obs = CircuitObserver(["χs" => linkdims, "χmax" => maxlinkdim, "pauliX2" => pauliX2, "pauliYs" => pauliYs])

  ψ = runcircuit(circuit; observer! = obs)
  @test haskey(obs.results,"χs") 
  @test haskey(obs.results,"χmax")
  @test haskey(obs.results,"pauliX2")
  @test haskey(obs.results,"pauliYs")
  @test length(obs.results["χs"]) == depth
  @test length(obs.results["χs"][1]) == N-1
  @test length(obs.results["χmax"]) == depth
  @test length(obs.results["pauliX2"]) == depth
  @test length(obs.results["pauliYs"]) == depth
  for d in 1:depth
    @test length(obs.results["pauliYs"][d]) == N
  end
end


@testset "tomography observer output" begin
  Random.seed!(1234)
  data,Ψ = readsamples("../examples/data/qst_circuit_test.h5")
  test_data = copy(data[1:10,:])
  N = length(Ψ)     # Number of qubits
  χ = maxlinkdim(Ψ) # Bond dimension of variational MPS
  ψ0 = randomstate(Ψ; χ = χ, σ = 0.1)
  opt = SGD(η = 0.01)
  obs = TomographyObserver()
  ψ = tomography(data, ψ0;
                 test_data = test_data,
                 optimizer = opt,
                 batchsize = 10,
                 epochs = 3,
                 target = Ψ,
                 observer! = obs)
  
  @test length(obs.fidelity) == 3
  @test length(obs.fidelity_bound) == 0
  @test length(obs.frobenius_distance) == 0
  @test length(obs.trace_preserving_distance) == 0
  @test length(obs.train_loss) == 3
  @test length(obs.test_loss) == 3
  
  data, ϱ = readsamples("../examples/data/qst_circuit_noisy_test.h5")
  test_data = copy(data[1:10,:])
  N = length(ϱ)     # Number of qubits
  χ = maxlinkdim(ϱ) # Bond dimension of variational LPDO
  ξ = 2             # Kraus dimension of variational LPDO
  ρ0 = randomstate(ϱ; mixed = true, χ = χ, ξ = ξ, σ = 0.1)
  opt = SGD(η = 0.01)
  obs = TomographyObserver()
  ρ = tomography(data, ρ0;
                 test_data = test_data,
                 optimizer = opt,
                 batchsize = 10,
                 epochs = 3,
                 target = ϱ,
                 observer! = obs)

  @test length(obs.fidelity) == 3
  @test length(obs.fidelity_bound) == 3
  @test length(obs.frobenius_distance) == 3
  @test length(obs.trace_preserving_distance) == 0
  @test length(obs.train_loss) == 3
  @test length(obs.test_loss) == 3
    
  data, U = readsamples("../examples/data/qpt_circuit_test.h5")
  test_data = copy(data[1:10,:])
  N = length(U)     # Number of qubits
  χ = maxlinkdim(U) # Bond dimension of variational MPS
  opt = SGD(η = 0.1)
  V0 = randomprocess(U; mixed = false, χ = χ)
  obs = TomographyObserver()
  V = tomography(data, V0;
                 test_data = test_data,
                 optimizer = opt,
                 batchsize = 10,
                 epochs = 3,
                 target = U,
                 observer! = obs)

  @test length(obs.fidelity) == 3
  @test length(obs.fidelity_bound) == 0
  @test length(obs.frobenius_distance) == 0
  @test length(obs.train_loss) == 3
  @test length(obs.test_loss) == 3
  @test length(obs.trace_preserving_distance) == 3
  
  # Noisy circuit
  Random.seed!(1234)
  data, ϱ = readsamples("../examples/data/qpt_circuit_noisy_test.h5")
  test_data = copy(data[1:10,:])
  N = length(ϱ)
  χ = 8
  ξ = 2
  Λ0 = randomprocess(ϱ; mixed = true, χ = χ, ξ = ξ, σ = 0.1)
  opt = SGD(η = 0.1)
  obs = TomographyObserver()
  Λ = tomography(data, Λ0;
                 test_data = test_data,
                 optimizer = opt,
                 mixed = true,
                 batchsize = 10,
                 epochs = 3,
                 target = ϱ,
                 observer! = obs)

  #@test length(obs.fidelity) == 3
  @test length(obs.fidelity_bound) == 3
  @test length(obs.frobenius_distance) == 3
  @test length(obs.train_loss) == 3
  @test length(obs.test_loss) == 3
  @test length(obs.trace_preserving_distance) == 3
end
