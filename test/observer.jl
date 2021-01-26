using PastaQ
using ITensors
using Random
using Test

@testset "measure one-body operators" begin

  N = 10
  depth = 10
  ψ = runcircuit(randomcircuit(N,depth))
  
  @test measure(ψ,("X",2)) ≈ inner(ψ,runcircuit(ψ,("X",2)))  
  @test measure(ψ,("Y",1)) ≈ inner(ψ,runcircuit(ψ,("Y",1))) 
  @test measure(ψ,("Z",4)) ≈ inner(ψ,runcircuit(ψ,("Z",4))) 
  
  results = measure(ψ,"X")
  for j in 1:length(ψ)
    @test results[j] ≈ inner(ψ,runcircuit(ψ,("X",j)))
  end

  results = measure(ψ,("X",1:2:length(ψ)))
  for j in 1:2:length(ψ)
    @test results[(j+1)÷2] ≈ inner(ψ,runcircuit(ψ,("X",j)))
  end

  results = measure(ψ,("Y",[1,3,5]))
  for j in [1,3,5]
    @test results[(j+1)÷2] ≈ inner(ψ,runcircuit(ψ,("Y",j)))
  end
end

@testset "circuitobserver constructor" begin

  # standard: inputs are named operations corresponding to gates
  obs = CircuitObserver("X(n)" => "X")
  @test haskey(obs.results,"X(n)")
  @test haskey(obs.measurements,"X(n)")
  obs = CircuitObserver("X1" => ("X",1))
  @test haskey(obs.results,"X1")
  @test haskey(obs.measurements,"X1")
  obs = CircuitObserver("X1:3" => ("X",1:3))
  @test haskey(obs.results,"X1:3")
  @test haskey(obs.measurements,"X1:3")
  
  obs = CircuitObserver(["X(n)" => "X", "Y1" => ("Y",1)])
  @test haskey(obs.results,"X(n)")
  @test haskey(obs.measurements,"X(n)")
  @test haskey(obs.results,"Y1")
  @test haskey(obs.measurements,"Y1")

  # unnamed inputs corresponding to gates
  obs = CircuitObserver("X")
  @test haskey(obs.results,"X(n)")
  @test haskey(obs.measurements,"X(n)")
  obs = CircuitObserver(("X",1))
  @test haskey(obs.results,"X1")
  @test haskey(obs.measurements,"X1")
  obs = CircuitObserver(("X",1:3))
  @test haskey(obs.results,"X1:3")
  @test haskey(obs.measurements,"X1:3")
  
  obs = CircuitObserver(["X",("Y",1)])
  @test haskey(obs.results,"X(n)")
  @test haskey(obs.measurements,"X(n)")
  @test haskey(obs.results,"Y1")
  @test haskey(obs.measurements,"Y1")

  # predefined functions 
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
  @test haskey(obs.measurements,"χs") 
  @test haskey(obs.measurements,"χmax")
  @test haskey(obs.measurements,"pauliX2")
  @test haskey(obs.measurements,"pauliYs")
  @test haskey(obs.results,"χs") 
  @test haskey(obs.results,"χmax")
  @test haskey(obs.results,"pauliX2")
  @test haskey(obs.results,"pauliYs")
  
  obs = CircuitObserver([linkdims, maxlinkdim, pauliX2, pauliYs])
  @test haskey(obs.measurements,"linkdims") 
  @test haskey(obs.measurements,"maxlinkdim")
  @test haskey(obs.measurements,"pauliX2")
  @test haskey(obs.measurements,"pauliYs")
  @test haskey(obs.results,"linkdims") 
  @test haskey(obs.results,"maxlinkdim")
  @test haskey(obs.results,"pauliX2")
  @test haskey(obs.results,"pauliYs")


  obs = CircuitObserver(["obs3" => norm, "obs1" => "X","obs2" => "Y"])
  @test haskey(obs.measurements,"obs1")
  @test haskey(obs.measurements,"obs2")
  @test haskey(obs.measurements,"obs3")
  @test haskey(obs.results,"obs1")
  @test haskey(obs.results,"obs2")
  @test haskey(obs.results,"obs3")
  obs = CircuitObserver([norm, "X"])
  @test haskey(obs.measurements,"norm")
  @test haskey(obs.measurements,"X(n)")
  @test haskey(obs.results,"norm")
  @test haskey(obs.results,"X(n)")

end

@testset "circuitobserver measurements: one-body" begin
  N = 10
  depth = 8
  circuit = randomcircuit(N, depth; seed = 1234)
  ψ = runcircuit(circuit)

  obs = CircuitObserver(["X","Y","Z"])
  PastaQ.measure!(obs,ψ)
  @test length(obs.results["X(n)"]) == 1
  @test length(obs.results["X(n)"][1]) == N
  @test length(obs.results["Y(n)"]) == 1
  @test length(obs.results["Y(n)"][1]) == N
  @test length(obs.results["Z(n)"]) == 1
  @test length(obs.results["Z(n)"][1]) == N

  obs = CircuitObserver(["X","Y","Z",norm,maxlinkdim])
  PastaQ.measure!(obs,ψ)
  @test length(obs.results["X(n)"]) == 1
  @test length(obs.results["X(n)"][1]) == N
  @test length(obs.results["Y(n)"]) == 1
  @test length(obs.results["Y(n)"][1]) == N
  @test length(obs.results["Z(n)"]) == 1
  @test length(obs.results["Z(n)"][1]) == N
  @test obs.results["norm"][1] ≈ norm(ψ)
  @test obs.results["maxlinkdim"][1] == maxlinkdim(ψ)
end

@testset "circuit observer" begin
  N = 6
  depth = 5
  R = 3
  Random.seed!(1234)
  circuit = Vector{Vector{<:Tuple}}(undef, depth)
  for d in 1:depth
    layer = Tuple[]
    bonds = PastaQ.randompairing(N,R)
    PastaQ.twoqubitlayer!(layer,"randU", bonds)
    circuit[d] = layer
  end
   
  obs = CircuitObserver("X")
  ψ = runcircuit(circuit; observer! = obs)
  @test length(obs.results["X(n)"]) == depth
  @test length(obs.results["X(n)"][1]) == N
  

  obs = CircuitObserver([("X",1:3),norm,maxlinkdim])
  runcircuit(circuit; observer! = obs)
  @test length(obs.results["X1:3"]) == depth
  @test length(obs.results["X1:3"][1]) == 3
  @test length(obs.results["norm"]) == depth 
  @test obs.results["norm"][end] ≈ norm(ψ) 
  @test obs.results["maxlinkdim"][end] ≈ maxlinkdim(ψ) 

end


#@testset "tomography observer output" begin
#  Random.seed!(1234)
#  data,Ψ = readsamples("../examples/data/qst_circuit_test.h5")
#  test_data = copy(data[1:10,:])
#  N = length(Ψ)     # Number of qubits
#  χ = maxlinkdim(Ψ) # Bond dimension of variational MPS
#  ψ0 = randomstate(Ψ; χ = χ, σ = 0.1)
#  opt = SGD(η = 0.01)
#  obs = TomographyObserver()
#  ψ = tomography(data, ψ0;
#                 test_data = test_data,
#                 optimizer = opt,
#                 batchsize = 10,
#                 epochs = 3,
#                 target = Ψ,
#                 observer! = obs)
#  
#  @test length(obs.fidelity) == 3
#  @test length(obs.fidelity_bound) == 0
#  @test length(obs.frobenius_distance) == 0
#  @test length(obs.trace_preserving_distance) == 0
#  @test length(obs.train_loss) == 3
#  @test length(obs.test_loss) == 3
#  
#  data, ϱ = readsamples("../examples/data/qst_circuit_noisy_test.h5")
#  test_data = copy(data[1:10,:])
#  N = length(ϱ)     # Number of qubits
#  χ = maxlinkdim(ϱ) # Bond dimension of variational LPDO
#  ξ = 2             # Kraus dimension of variational LPDO
#  ρ0 = randomstate(ϱ; mixed = true, χ = χ, ξ = ξ, σ = 0.1)
#  opt = SGD(η = 0.01)
#  obs = TomographyObserver()
#  ρ = tomography(data, ρ0;
#                 test_data = test_data,
#                 optimizer = opt,
#                 batchsize = 10,
#                 epochs = 3,
#                 target = ϱ,
#                 observer! = obs)
#
#  @test length(obs.fidelity) == 3
#  @test length(obs.fidelity_bound) == 3
#  @test length(obs.frobenius_distance) == 3
#  @test length(obs.trace_preserving_distance) == 0
#  @test length(obs.train_loss) == 3
#  @test length(obs.test_loss) == 3
#    
#  data, U = readsamples("../examples/data/qpt_circuit_test.h5")
#  test_data = copy(data[1:10,:])
#  N = length(U)     # Number of qubits
#  χ = maxlinkdim(U) # Bond dimension of variational MPS
#  opt = SGD(η = 0.1)
#  V0 = randomprocess(U; mixed = false, χ = χ)
#  obs = TomographyObserver()
#  V = tomography(data, V0;
#                 test_data = test_data,
#                 optimizer = opt,
#                 batchsize = 10,
#                 epochs = 3,
#                 target = U,
#                 observer! = obs)
#
#  @test length(obs.fidelity) == 3
#  @test length(obs.fidelity_bound) == 0
#  @test length(obs.frobenius_distance) == 0
#  @test length(obs.train_loss) == 3
#  @test length(obs.test_loss) == 3
#  @test length(obs.trace_preserving_distance) == 3
#  
#  # Noisy circuit
#  Random.seed!(1234)
#  data, ϱ = readsamples("../examples/data/qpt_circuit_noisy_test.h5")
#  test_data = copy(data[1:10,:])
#  N = length(ϱ)
#  χ = 8
#  ξ = 2
#  Λ0 = randomprocess(ϱ; mixed = true, χ = χ, ξ = ξ, σ = 0.1)
#  opt = SGD(η = 0.1)
#  obs = TomographyObserver()
#  Λ = tomography(data, Λ0;
#                 test_data = test_data,
#                 optimizer = opt,
#                 mixed = true,
#                 batchsize = 10,
#                 epochs = 3,
#                 target = ϱ,
#                 observer! = obs)
#
#  #@test length(obs.fidelity) == 3
#  @test length(obs.fidelity_bound) == 3
#  @test length(obs.frobenius_distance) == 3
#  @test length(obs.train_loss) == 3
#  @test length(obs.test_loss) == 3
#  @test length(obs.trace_preserving_distance) == 3
#end
