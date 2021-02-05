using PastaQ
using ITensors
using Random
using Test

@testset "circuitobserver constructor" begin

  ## standard: inputs are named operations corresponding to gates
  obs = Observer("X(n)" => "X")
  @test haskey(obs.measurements,"X(n)")
  @test PastaQ.measurement(obs,"X(n)") == "X"
  @test results(obs,"X(n)") == [] 
  obs = Observer("X1" => ("X",1))
  @test haskey(obs.measurements,"X1")
  @test PastaQ.measurement(obs,"X1") == ("X",1)
  @test results(obs,"X1") == [] 
  obs = Observer("X1:3" => ("X",1:3))
  @test haskey(obs.measurements,"X1:3")
  @test PastaQ.measurement(obs,"X1:3") == ("X",1:3)
  @test results(obs,"X1:3") == [] 
  
  obs = Observer(["X(n)" => "X", "Y1" => ("Y",1)])
  @test haskey(obs.measurements,"X(n)")
  @test PastaQ.measurement(obs,"X(n)") == "X"
  @test results(obs,"X(n)") == [] 
  @test haskey(obs.measurements,"Y1")
  @test PastaQ.measurement(obs,"Y1") == ("Y",1)
  @test results(obs,"Y1") == [] 

  # unnamed inputs corresponding to gates
  obs = Observer("X")
  @test haskey(obs.measurements,"X")
  @test PastaQ.measurement(obs,"X") == "X"
  @test results(obs,"X") == []
  obs = Observer(("X",1))
  @test haskey(obs.measurements,"X(1)")
  @test PastaQ.measurement(obs,"X(1)") == ("X",1)
  @test results(obs,"X(1)") == []
  obs = Observer(("X",1:3))
  @test haskey(obs.measurements,"X(1:3)")
  @test PastaQ.measurement(obs,"X(1:3)") == ("X",1:3)
  @test results(obs,"X(1:3)") == [] 
  
  obs = Observer(["X",("Y",1)])
  @test haskey(obs.measurements,"X")
  @test haskey(obs.measurements,"Y(1)")
  @test PastaQ.measurement(obs,"X") == "X"
  @test results(obs,"X") == []
  @test PastaQ.measurement(obs,"Y(1)") == ("Y",1)
  @test results(obs,"Y(1)") == []

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
  @test PastaQ.measurement(obs,"χs") == linkdims
  @test PastaQ.measurement(obs,"χmax") == maxlinkdim
  @test PastaQ.measurement(obs,"pauliX2") == pauliX2
  @test PastaQ.measurement(obs,"pauliYs") == pauliYs
  @test results(obs,"χs") == []  
  @test results(obs,"χmax") == []
  @test results(obs,"pauliX2") == []
  @test results(obs,"pauliYs") == [] 
  
  obs = Observer([linkdims, maxlinkdim, pauliX2, pauliYs])
  @test haskey(obs.measurements,"linkdims") 
  @test haskey(obs.measurements,"maxlinkdim")
  @test haskey(obs.measurements,"pauliX2")
  @test haskey(obs.measurements,"pauliYs")
  @test PastaQ.measurement(obs,"linkdims") == linkdims
  @test PastaQ.measurement(obs,"maxlinkdim") == maxlinkdim
  @test PastaQ.measurement(obs,"pauliX2") == pauliX2
  @test PastaQ.measurement(obs,"pauliYs") == pauliYs
  @test results(obs,"linkdims") == []  
  @test results(obs,"maxlinkdim") == []
  @test results(obs,"pauliX2") == []
  @test results(obs,"pauliYs") == [] 

  obs = Observer(["obs1" => norm, "obs2" => "X","obs3" => "Y"])
  @test haskey(obs.measurements,"obs1")
  @test haskey(obs.measurements,"obs2")
  @test haskey(obs.measurements,"obs3")
  @test PastaQ.measurement(obs,"obs1") == norm
  @test PastaQ.measurement(obs,"obs2") == "X"
  @test PastaQ.measurement(obs,"obs3") == "Y"
  @test results(obs,"obs1") == [] 
  @test results(obs,"obs2") == [] 
  @test results(obs,"obs3") == [] 

  obs = Observer([norm, "X"])
  @test haskey(obs.measurements,"norm")
  @test haskey(obs.measurements,"X")

  obs = Observer()
  push!(obs,"obs3" => norm)
  push!(obs,"obs1" => "X","obs2" => "Y")
  @test haskey(obs.measurements,"obs1")
  @test haskey(obs.measurements,"obs2")
  @test haskey(obs.measurements,"obs3")
  @test PastaQ.measurement(obs, "obs1") == "X"
  @test PastaQ.measurement(obs, "obs2") == "Y"
  @test results(obs,"obs1") == [] 
  @test results(obs,"obs2") == [] 

  obs = Observer()
  push!(obs,norm)
  push!(obs,norm, "X")
  @test haskey(obs.measurements,"X")
  @test PastaQ.measurement(obs, "X") == "X"
  @test PastaQ.measurement(obs, "norm") == norm
  @test results(obs,"X") == []
  @test results(obs,"norm") == []
  @test haskey(obs.measurements,"norm")
  f1(ψ::MPS) = norm(ψ) * a1
  f2(ψ::MPS) = sqrt(a2)*norm(ψ)*a1
  
  obs = Observer(f1)
  push!(obs,f2) 
  @test haskey(obs.measurements,"f1")
  @test haskey(obs.measurements,"f2")
  @test PastaQ.measurement(obs, "f1") == f1 
  @test PastaQ.measurement(obs, "f2") == f2
  @test results(obs,"f1") == []
  @test results(obs,"f2") == []
end


@testset "circuitobserver measurements: one-body" begin
  N = 10
  depth = 8
  circuit = randomcircuit(N, depth)
  ψ = runcircuit(circuit)

  obs = Observer(["X","Y","Z"])
  PastaQ.measure!(obs,ψ)
  @test length(last(obs.measurements["X"])) == 1
  @test length(last(obs.measurements["X"])[1]) == N
  @test length(last(obs.measurements["Y"])) == 1
  @test length(last(obs.measurements["Y"])[1]) == N
  @test length(last(obs.measurements["Z"])) == 1
  @test length(last(obs.measurements["Z"])[1]) == N

  obs = Observer(["X","Y","Z",norm,maxlinkdim])
  PastaQ.measure!(obs,ψ)
  @test length(last(obs.measurements["X"])) == 1
  @test length(last(obs.measurements["X"])[1]) == N
  @test length(last(obs.measurements["Y"])) == 1
  @test length(last(obs.measurements["Y"])[1]) == N
  @test length(last(obs.measurements["Z"])) == 1
  @test length(last(obs.measurements["Z"])[1]) == N
  @test results(obs,"norm")[end] ≈ norm(ψ)
  @test results(obs,"maxlinkdim")[end] ≈ maxlinkdim(ψ)
end

@testset "circuit observer" begin
  N = 6
  depth = 5
  R = 3
  Random.seed!(1234)
  circuit = Vector{Vector{<:Any}}(undef, depth)
  for d in 1:depth
    layer = []
    layer = [("CX",(1,rand(2:N))),("CX",(1,rand(2:N))),("CX",(1,rand(2:N)))]#gatelayer(bonds,"CX") 
    circuit[d] = layer
  end
  
  obs = Observer("X")
  ψ = runcircuit(circuit; observer! = obs)
  @test length(results(obs,"X")) == depth
  @test length(results(obs,"X")[1]) == N
  

  obs = Observer([("X",1:3),norm,maxlinkdim])
  ψ = runcircuit(circuit; observer! = obs,  move_sites_back_before_measurements = true)
  @test length(results(obs,"X(1:3)")) == depth
  @test length(results(obs,"X(1:3)")[end]) == 3
  @test length(results(obs,"norm")) == depth 
  @test results(obs,"norm")[end] ≈ norm(ψ)
  @test results(obs,"maxlinkdim")[end] ≈ maxlinkdim(ψ)
  ϕ = qubits(ψ) 
  f1(ψ::MPS) = fidelity(ψ,ϕ)
  f2(ψ::MPS) = fidelity_bound(ψ,ϕ) 
  obs = Observer(f1)
  push!(obs,f2) 
  ψ = runcircuit(circuit; observer! = obs)
  @test length(results(obs,"f1")) == depth
  @test length(results(obs,"f2")) == depth
  @test results(obs,"f1")[end] == fidelity(ψ,ϕ)
  @test results(obs,"f2")[end] == fidelity_bound(ψ,ϕ) 
end


@testset "state tomography observer output" begin
  Random.seed!(1234)
  data,Ψ = readsamples("../examples/data/qst_circuit_test.h5")
  test_data = copy(data[1:10,:])
  N = length(Ψ)     # Number of qubits
  χ = maxlinkdim(Ψ) # Bond dimension of variational MPS
  ψ0 = randomstate(Ψ; χ = χ, σ = 0.1)
  opt = SGD(η = 0.01)
  
  F(ψ::MPS) = fidelity(ψ,Ψ)
  obs = Observer([maxlinkdim,norm,("X",1),F])
  epochs = 18
  
  obs2 = copy(obs)
  batchsize = 10
  measurement_frequency = 3

  ψ = tomography(data, ψ0;
                 test_data = test_data,
                 batchsize = 10,
                 epochs = epochs,
                 observer! = obs,
                 measurement_frequency = measurement_frequency,
                 print_metrics = false)
  
  @test haskey(obs.measurements,"X(1)")
  @test haskey(obs.measurements,"maxlinkdim")
  @test haskey(obs.measurements,"F")
  @test haskey(obs.measurements,"test_loss")
  @test haskey(obs.measurements,"train_loss")
  @test length(results(obs,"X(1)")) == epochs ÷measurement_frequency
  @test length(results(obs,"maxlinkdim")) == epochs÷measurement_frequency
  @test length(results(obs,"F"))   == epochs÷measurement_frequency
  @test length(results(obs,"test_loss"))  == epochs÷measurement_frequency
  @test length(results(obs,"train_loss")) == epochs÷measurement_frequency

  params = results(obs,"parameters") 
  @test params["batchsize"] == 10
  @test params["measurement_frequency"] == 3
  @test params["dataset_size"] == size(data,1)
  @test haskey(params,"SGD")
  @test haskey(params["SGD"],:η)
  @test haskey(params["SGD"],:γ)
  @test params["SGD"][:η] == 0.01

end

@testset "process tomography observer output" begin
  Random.seed!(1234)
  data,V = readsamples("../examples/data/qpt_circuit_test.h5")
  test_data = copy(data[1:10,:])
  N = length(V)     # Number of qubits
  χ = maxlinkdim(V) # Bond dimension of variational MPS
  U0 = randomprocess(V; χ = χ, σ = 0.1)
  opt = SGD(η = 0.01)
  
  F(U::MPO) = fidelity(U,V; process = true)
  obs = Observer([maxlinkdim,norm,F])
  epochs = 9
  
  obs2 = copy(obs)
  batchsize = 10
  measurement_frequency = 3

  U = tomography(data, U0;
                 test_data = test_data,
                 batchsize = 10,
                 epochs = epochs,
                 observer! = obs,
                 measurement_frequency = measurement_frequency,
                 print_metrics = false)
  
  @test haskey(obs.measurements,"maxlinkdim")
  @test haskey(obs.measurements,"F")
  @test haskey(obs.measurements,"test_loss")
  @test haskey(obs.measurements,"train_loss")
  @test length(results(obs,"maxlinkdim")) == epochs÷measurement_frequency
  @test length(results(obs,"F"))   == epochs÷measurement_frequency
  @test length(results(obs,"test_loss"))  == epochs÷measurement_frequency
  @test length(results(obs,"train_loss")) == epochs÷measurement_frequency

  params = results(obs,"parameters") 
  @test params["batchsize"] == 10
  @test params["measurement_frequency"] == 3
  @test params["dataset_size"] == size(data,1)
  @test haskey(params,"SGD")
  @test haskey(params["SGD"],:η)
  @test haskey(params["SGD"],:γ)
  @test params["SGD"][:η] == 0.01

end
