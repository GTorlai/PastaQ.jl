using PastaQ
using PastaQ.ITensors
using Test
using LinearAlgebra
using Random
using Observers
using JLD2
using Optimisers

@testset "write and read samples" begin
  N = 3
  depth = 4
  nshots = 100
  circuit = randomcircuit(N, depth)
  path = "test_data_writesamples.h5"
  
  data, X = getsamples(circuit, nshots; local_basis = nothing) 
  writesamples(data, path)
  datatest = readsamples(path)
  @test datatest ≈ data
  writesamples(data, X, path)
  datatest, Xtest = readsamples(path)
  @test X ≈ Xtest
  @test datatest ≈ data
  
  data, X = getsamples(circuit, nshots; local_basis = nothing, noise = ("DEP",(p=0.01,))) 
  writesamples(data, path)
  datatest = readsamples(path)
  @test datatest ≈ data
  writesamples(data, X, path)
  datatest, Xtest = readsamples(path)
  @test X ≈ Xtest
  @test datatest ≈ data
  
  data, _ = getsamples(circuit, nshots; local_basis = nothing, noise = ("DEP",(p=0.01,))) 
  X = randomstate(N; ξ = 2, χ = 3)
  writesamples(data, path)
  datatest = readsamples(path)
  @test datatest ≈ data
  writesamples(data, X, path)
  datatest, Xtest = readsamples(path)
  @test X.X ≈ Xtest.X
  @test datatest ≈ data

  
  data, X = getsamples(circuit, nshots; local_basis = ["X","Y","Z"]) 
  writesamples(data, path)
  datatest = readsamples(path)
  @test all(data .== datatest)
  writesamples(data, X, path)
  datatest, Xtest = readsamples(path)
  @test X ≈ Xtest
  @test all(data .== datatest)
  
  data, X = getsamples(circuit, nshots; local_basis = ["X","Y","Z"], noise = ("DEP",(p=0.01,))) 
  writesamples(data, path)
  datatest = readsamples(path)
  @test all(data .== datatest)
  writesamples(data, X, path)
  datatest, Xtest = readsamples(path)
  @test X ≈ Xtest
  @test all(data .== datatest)
  
  data, _ = getsamples(circuit, nshots; local_basis = ["X","Y","Z"], noise = ("DEP",(p=0.01,))) 
  X = randomstate(N; ξ = 2, χ = 3)
  writesamples(data, path)
  datatest = readsamples(path)
  @test all(data .== datatest)
  writesamples(data, X, path)
  datatest, Xtest = readsamples(path)
  @test X.X ≈ Xtest.X
  @test all(data .== datatest)


  data, X = getsamples(circuit, nshots; local_basis = ["X","Y","Z"], process = true) 
  writesamples(data, path)
  datatest = readsamples(path)
  @test all(data .== datatest)
  writesamples(data, X, path)
  datatest, Xtest = readsamples(path)
  @test X ≈ Xtest
  @test all(data .== datatest)
  
  data, _ = getsamples(circuit, nshots; local_basis = ["X","Y","Z"],process = true, noise = ("DEP",(p=0.01,))) 
  X = randomprocess(N; ξ = 2, χ = 3)
  writesamples(data, path)
  datatest = readsamples(path)
  @test all(data .== datatest)
  writesamples(data, X, path)
  datatest, Xtest = readsamples(path)
  @test X.X ≈ Xtest.X
  @test all(data .== datatest)

end

@testset "circuit observer: MPS" begin
  N = 8
  depth = 9
  R = 5
  Random.seed!(1234)
  #circuit = Vector{Vector{<:Any}}(undef, depth)
  circuit = randomcircuit(N, depth)
  layer = Tuple[]
  push!(circuit, [("CX",(1,N)),])

  sites = siteinds("Qubit", N)
  
  outputpath = "simulation"
  ϕ = randomstate(sites; χ = 10, normalize=true)
  ψ = runcircuit(sites, circuit)
  Ftest = fidelity(ψ,ϕ)
  f(ψ::MPS; kwargs...) = fidelity(ψ, ϕ)#; kwargs...) = fidelity(ψ, ϕ)
  obs = Observer(["f" => f])
  ψ = runcircuit(sites, circuit; (observer!)=obs, move_sites_back_before_measurements=true,
                 outputpath = outputpath, savestate = true, outputlevel = 0)
  @test Ftest ≈ results(obs, "f")[end]
  @test length(results(obs, "f")) == depth+1
  
  obs2 = load("simulation_observer.jld2")
  for (k,v) in obs2
    @test last(v) ≈ results(obs, k)
  end

  fin = h5open("simulation_state.h5","r")
  M = read(fin, "state", MPS)
  close(fin)
  @test M ≈ ψ
end

@testset "circuit observer: MPO" begin
  N = 4
  depth = 5
  #circuit = Vector{Vector{<:Any}}(undef, depth)
  sites = siteinds("Qubit", N)
  circuit = randomcircuit(N, depth)
  
  @disable_warn_order begin
    L = randomstate(sites; χ = 10, ξ = 3, normalize=true)
    ϱ = MPO(L)
    ρ = runcircuit(sites, circuit; noise = ("DEP",(p=0.001,)))
    Ftest = fidelity(ϱ,ρ)
    g(ρ::MPO; kwargs...) = fidelity(ρ, ϱ)#; kwargs...) = fidelity(ψ, ϕ)
    obs = Observer(["g" => g])
    outputpath = "simulation"
    ρ = runcircuit(sites, circuit; 
                   (observer!)=obs, 
                   move_sites_back_before_measurements=true, 
                   outputlevel = 0,
                   noise = ("DEP",(p=0.001,)),
                   outputpath = outputpath, savestate = true)
 end
  @test Ftest ≈ results(obs, "g")[end]
  @test length(results(obs, "g")) == depth
  obs2 = load("simulation_observer.jld2")
  for (k,v) in obs2
    @test last(v) ≈ results(obs, k)
  end
  
  fin = h5open("simulation_state.h5","r")
  M = read(fin, "state", MPO)
  close(fin)
  @test M ≈ ρ
end

@testset "state tomography observer output: MPS" begin
  Random.seed!(1234)
  N = 4
  depth = 4
  nshots = 100
  circuit = randomcircuit(N, depth)
  data, Ψ = getsamples(circuit, nshots; local_basis=["X", "Y", "Z"])
  test_data = copy(data[1:10, :])
  
  N = length(Ψ)     # Number of productstate
  χ = maxlinkdim(Ψ) # Bond dimension of variational MPS
  ψ0 = randomstate(Ψ; χ=χ, σ=0.1)
  opt = Optimisers.Descent(0.01)

  F(ψ::MPS; kwargs...) = fidelity(ψ, Ψ)
  obs = Observer(["F" => F])
  #obs = Observer([maxlinkdim, norm, ("X", 1), F])
  epochs = 18

  batchsize = 10
  observe_step = 3

  outputpath = "simulation"
  ψ = tomography(
    data,
    ψ0;
    test_data=test_data,
    batchsize=10,
    epochs=epochs,
    (observer!)=obs,
    observe_step = observe_step,
    outputpath = outputpath,
    savestate = true,
    outputlevel = 0
   )
  @test length(results(obs, "F")) == epochs ÷ observe_step 

  obs2 = load("simulation_observer.jld2")
  for (k,v) in obs2
    @test last(v) ≈ results(obs, k)
  end
  
  fin = h5open("simulation_state.h5","r")
  M = read(fin, "state", MPS)
  close(fin)
  @test M ≈ ψ
end

@testset "state tomography observer output: LPDO" begin
  Random.seed!(1234)
  N = 4
  depth = 4
  nshots = 100
  circuit = randomcircuit(N, depth)
  data, ϱ = getsamples(circuit, nshots; 
                       local_basis=["X", "Y", "Z"], 
                       noise = ("DEP",(p=0.01,))) 
  test_data = copy(data[1:10, :])
  
  N = length(ϱ)     # Number of productstate
  χ = maxlinkdim(ϱ) # Bond dimension of variational MPS
  ρ = randomstate(ϱ; χ=χ÷2, ξ=2)
  opt = Optimisers.Descent(0.01)

  F(ρ::LPDO; kwargs...) = fidelity(ρ, ϱ)
  obs = Observer(["F" => F])
  epochs = 9

  batchsize = 10
  observe_step = 3

  outputpath = "simulation"
  ρ = tomography(
    data,
    ρ;
    test_data=test_data,
    batchsize=10,
    epochs=epochs,
    (observer!)=obs,
    observe_step = observe_step,
    outputpath = outputpath,
    savestate = true,
    outputlevel = 0
  )
  @test length(results(obs, "F")) == epochs ÷ observe_step 

  obs2 = load("simulation_observer.jld2")
  for (k,v) in obs2
    @test last(v) ≈ results(obs, k)
  end
  
  fin = h5open("simulation_state.h5","r")
  M = read(fin, "state", LPDO{MPO})
  close(fin)
  @test M.X ≈ ρ.X
end


@testset "process tomography observer output: MPO" begin
  Random.seed!(1234)
  N = 3
  depth = 4
  nshots = 100
  circuit = randomcircuit(N, depth)
  data, V = getsamples(circuit, nshots; local_basis=["X", "Y", "Z"], process = true)
  test_data = copy(data[1:10, :])
  N = length(V)     # Number of productstate
  χ = maxlinkdim(V) # Bond dimension of variational MPS
  U0 = randomprocess(V; χ=χ, σ=0.1)
  opt = Optimisers.Descent(0.01)

  F(U::MPO; kwargs...) = fidelity(U, V; process=true)
  obs = Observer(["F" => F])
  epochs = 9

  batchsize = 10
  observe_step = 3

  outputpath = "simulation"
  U = tomography(
    data,
    U0;
    test_data=test_data,
    batchsize=10,
    epochs=epochs,
    (observer!)=obs,
    observe_step = observe_step,
    outputpath = outputpath,
    savestate = true,
    outputlevel = 0
   )

  @test length(results(obs, "F")) == epochs ÷ observe_step 

  obs2 = load("simulation_observer.jld2")
  for (k,v) in obs2
    @test last(v) ≈ results(obs, k)
  end
  fin = h5open("simulation_state.h5","r")
  M = read(fin, "state", MPO)
  close(fin)
  @test M ≈ U
end



@testset "state tomography observer output: LPDO" begin
  Random.seed!(1234)
  N = 2
  depth = 4
  nshots = 100
  circuit = randomcircuit(N, depth)
  data, Φ = getsamples(circuit, nshots; 
                       local_basis=["X", "Y", "Z"], 
                       noise = ("DEP",(p=0.01,)), 
                       process = true)
  test_data = copy(data[1:10, :])
  
  N = length(Φ)     # Number of productstate
  χ = maxlinkdim(Φ) # Bond dimension of variational MPS
  Λ = randomprocess(Φ; χ=χ÷2, ξ=2)
  opt = Optimisers.Descent(0.01)

  F(Λ::LPDO; kwargs...) = fidelity(Λ, Φ)
  obs = Observer(["F" => F])
  epochs = 9

  batchsize = 10
  observe_step = 3

  outputpath = "simulation"
  Λ = tomography(
    data,
    Λ;
    test_data=test_data,
    batchsize=10,
    epochs=epochs,
    (observer!)=obs,
    observe_step = observe_step,
    outputpath = outputpath,
    savestate = true,
    outputlevel = 0
  )
  @test length(results(obs, "F")) == epochs ÷ observe_step 

  obs2 = load("simulation_observer.jld2")
  for (k,v) in obs2
    @test last(v) ≈ results(obs, k)
  end
  
  fin = h5open("simulation_state.h5","r")
  M = read(fin, "state", LPDO{MPO})
  close(fin)
  @test M.X ≈ Λ.X
end

