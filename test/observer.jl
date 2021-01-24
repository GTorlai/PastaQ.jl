using PastaQ
using ITensors
using Random
using Test

@testset "circuit observer" begin
  N = 10
  depth = 8
  circuit = randomcircuit(N,8; layered = true, seed = 1234)

  function f1(ψ::MPS, site::Int)
    return norm(ψ[1])
  end
  
  observables = (f = f1, name = "f1", sites = 1:N)
  obs = Dict()
  ψ = runcircuit(circuit; observer! = obs, observables = observables)
  @show keys(obs)
  @test haskey(obs,"f1") 
  @test haskey(obs,"χ")
  @test haskey(obs,"χmax")
  @test length(obs["χ"]) == depth
  @test length(obs["χ"][1]) == N-1
  @test length(obs["χmax"]) == depth
  @test length(obs["f1"]) == depth
  @test length(obs["f1"][1]) == N
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
