using PastaQ
using ITensors
using Random
using Test

@testset "circuitobserver constructor" begin

  # standard: inputs are named operations corresponding to gates
  obs = Observer("X(n)" => "X")
  @test haskey(obs.results,"X(n)")
  @test haskey(obs.measurements,"X(n)")
  obs = Observer("X1" => ("X",1))
  @test haskey(obs.results,"X1")
  @test haskey(obs.measurements,"X1")
  obs = Observer("X1:3" => ("X",1:3))
  @test haskey(obs.results,"X1:3")
  @test haskey(obs.measurements,"X1:3")
  
  obs = Observer(["X(n)" => "X", "Y1" => ("Y",1)])
  @test haskey(obs.results,"X(n)")
  @test haskey(obs.measurements,"X(n)")
  @test haskey(obs.results,"Y1")
  @test haskey(obs.measurements,"Y1")

  # unnamed inputs corresponding to gates
  obs = Observer("X")
  @test haskey(obs.results,"X")
  @test haskey(obs.measurements,"X")
  obs = Observer(("X",1))
  @test haskey(obs.results,"X(1)")
  @test haskey(obs.measurements,"X(1)")
  obs = Observer(("X",1:3))
  @test haskey(obs.results,"X(1:3)")
  @test haskey(obs.measurements,"X(1:3)")
  
  obs = Observer(["X",("Y",1)])
  @test haskey(obs.results,"X")
  @test haskey(obs.measurements,"X")
  @test haskey(obs.results,"Y(1)")
  @test haskey(obs.measurements,"Y(1)")

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
  
  obs = Observer(["χs" => linkdims, "χmax" => maxlinkdim, "pauliX2" => pauliX2, "pauliYs" => pauliYs])
  @test haskey(obs.measurements,"χs") 
  @test haskey(obs.measurements,"χmax")
  @test haskey(obs.measurements,"pauliX2")
  @test haskey(obs.measurements,"pauliYs")
  @test haskey(obs.results,"χs") 
  @test haskey(obs.results,"χmax")
  @test haskey(obs.results,"pauliX2")
  @test haskey(obs.results,"pauliYs")
  
  obs = Observer([linkdims, maxlinkdim, pauliX2, pauliYs])
  @test haskey(obs.measurements,"linkdims") 
  @test haskey(obs.measurements,"maxlinkdim")
  @test haskey(obs.measurements,"pauliX2")
  @test haskey(obs.measurements,"pauliYs")
  @test haskey(obs.results,"linkdims") 
  @test haskey(obs.results,"maxlinkdim")
  @test haskey(obs.results,"pauliX2")
  @test haskey(obs.results,"pauliYs")


  obs = Observer(["obs3" => norm, "obs1" => "X","obs2" => "Y"])
  @test haskey(obs.measurements,"obs1")
  @test haskey(obs.measurements,"obs2")
  @test haskey(obs.measurements,"obs3")
  @test haskey(obs.results,"obs1")
  @test haskey(obs.results,"obs2")
  @test haskey(obs.results,"obs3")
  obs = Observer([norm, "X"])
  @test haskey(obs.measurements,"norm")
  @test haskey(obs.measurements,"X")
  @test haskey(obs.results,"norm")
  @test haskey(obs.results,"X")

  obs = Observer()
  push!(obs,"obs3" => norm)
  push!(obs,"obs1" => "X","obs2" => "Y")
  @test haskey(obs.measurements,"obs1")
  @test haskey(obs.measurements,"obs2")
  @test haskey(obs.measurements,"obs3")
  @test haskey(obs.results,"obs1")
  @test haskey(obs.results,"obs2")
  @test haskey(obs.results,"obs3")
  obs = Observer()
  push!(obs,norm, "X")
  @test haskey(obs.measurements,"norm")
  @test haskey(obs.measurements,"X")
  @test haskey(obs.results,"norm")
  @test haskey(obs.results,"X")

end

@testset "circuitobserver measurements: one-body" begin
  N = 10
  depth = 8
  circuit = randomcircuit(N, depth)
  ψ = runcircuit(circuit)

  obs = Observer(["X","Y","Z"])
  PastaQ.measure!(obs,ψ)
  @test length(obs.results["X"]) == 1
  @test length(obs.results["X"][1]) == N
  @test length(obs.results["Y"]) == 1
  @test length(obs.results["Y"][1]) == N
  @test length(obs.results["Z"]) == 1
  @test length(obs.results["Z"][1]) == N

  obs = Observer(["X","Y","Z",norm,maxlinkdim])
  PastaQ.measure!(obs,ψ)
  @test length(obs.results["X"]) == 1
  @test length(obs.results["X"][1]) == N
  @test length(obs.results["Y"]) == 1
  @test length(obs.results["Y"][1]) == N
  @test length(obs.results["Z"]) == 1
  @test length(obs.results["Z"][1]) == N
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
   
  obs = Observer("X")
  ψ = runcircuit(circuit; observer! = obs)
  @test length(obs.results["X"]) == depth
  @test length(obs.results["X"][1]) == N
  

  obs = Observer([("X",1:3),norm,maxlinkdim])
  runcircuit(circuit; observer! = obs)
  @test length(obs.results["X(1:3)"]) == depth
  @test length(obs.results["X(1:3)"][1]) == 3
  @test length(obs.results["norm"]) == depth 
  @test obs.results["norm"][end] ≈ norm(ψ) 
  @test obs.results["maxlinkdim"][end] ≈ maxlinkdim(ψ) 

end


@testset "tomography observer output" begin
  Random.seed!(1234)
  data,Ψ = readsamples("../examples/data/qst_circuit_test.h5")
  test_data = copy(data[1:10,:])
  N = length(Ψ)     # Number of qubits
  χ = maxlinkdim(Ψ) # Bond dimension of variational MPS
  ψ0 = randomstate(Ψ; χ = χ, σ = 0.1)
  opt = SGD(η = 0.01)
  
  fid(ψ::MPS) = fidelity(ψ,ψ0)
  obs = Observer([maxlinkdim,norm,("X",1),fid])
  epochs = 12
  ψ = tomography(data, ψ0;
                 test_data = test_data,
                 batchsize = 10,
                 epochs = epochs,
                 observer! = obs,
                 print_metrics = false)
  
  @test haskey(obs.results,"X(1)")
  @test length(obs.results["X(1)"]) == epochs
  @test haskey(obs.results,"maxlinkdim")
  @test length(obs.results["maxlinkdim"]) == epochs
  @test haskey(obs.results,"fid")
  @test length(obs.results["fid"]) == epochs
  @test haskey(obs.results,"test_loss")
  @test length(obs.results["test_loss"]) == epochs
  @test haskey(obs.results,"train_loss")
  @test length(obs.results["train_loss"]) == epochs
  @test haskey(obs.results,"simulation_time")
  @test haskey(obs.results,"parameters") 

  params = obs.results["parameters"]
  @test params["batchsize"] == 10
  @test params["measurement_frequency"] == 1
  @test haskey(params,"SGD")
  @test haskey(params["SGD"],:η)
  @test haskey(params["SGD"],:γ)
  @test params["SGD"][:η] == 0.01
  @show obs.results
end
