using PastaQ
using PastaQ.ITensors
using Test
using LinearAlgebra
using Random
using Observers
using JLD2
using HDF5
using Optimisers

# create a data/ directory to store results of testing, so they are not accidentally included in the git. 
# This repo is already excluded in the gitignore

Base.mkpath("data/")
prefix = "data"

@testset "write and read samples" begin
  N = 4
  depth = 2
  nshots = 100
  circuit = randomcircuit(N; depth=depth, twoqubitgates="CX", onequbitgates="Rx")
  path = prefix*"/"*"test_data_writesamples.h5"

  X = runcircuit(circuit)
  data = getsamples(X, nshots)
  writesamples(data, path)
  datatest = readsamples(path)
  @test datatest ≈ data
  writesamples(data, X, path)
  datatest, Xtest = readsamples(path)
  @test X ≈ Xtest
  @test datatest ≈ data

  X = runcircuit(circuit; noise=("DEP", (p=0.01,)))
  data = getsamples(X, nshots)
  writesamples(data, path)
  datatest = readsamples(path)
  @test datatest ≈ data
  writesamples(data, X, path)
  datatest, Xtest = readsamples(path)
  @test X ≈ Xtest
  @test datatest ≈ data

  X = randomstate(N; ξ=2, χ=3)
  writesamples(data, X, path)
  datatest, Xtest = readsamples(path)
  @test X.X ≈ Xtest.X
  @test datatest ≈ data

  X = runcircuit(circuit)
  bases = randombases(N, 10)
  data = getsamples(X, bases, nshots)
  writesamples(data, path)
  datatest = readsamples(path)
  @test all(data .== datatest)
  writesamples(data, X, path)
  datatest, Xtest = readsamples(path)
  @test X ≈ Xtest
  @test all(data .== datatest)

  X = runcircuit(circuit; noise=("DEP", (p=0.01,)))
  bases = randombases(N, 10)
  data = getsamples(X, bases, nshots)
  writesamples(data, path)
  datatest = readsamples(path)
  @test all(data .== datatest)
  writesamples(data, X, path)
  datatest, Xtest = readsamples(path)
  @test X ≈ Xtest
  @test all(data .== datatest)

  X = randomstate(N; ξ=2, χ=3)
  writesamples(data, X, path)
  datatest, Xtest = readsamples(path)
  @test X.X ≈ Xtest.X
  @test all(data .== datatest)

  X = runcircuit(circuit; process=true)
  preps = randompreparations(N, 8)
  bases = randombases(N, 10)
  data = getsamples(X, preps, bases, nshots)
  writesamples(data, path)
  datatest = readsamples(path)
  @test all(data .== datatest)
  writesamples(data, X, path)
  datatest, Xtest = readsamples(path)
  @test X ≈ Xtest
  @test all(data .== datatest)

  X = runcircuit(circuit; noise=("DEP", (p=0.01,)), process=true)
  preps = randompreparations(N, 8)
  bases = randombases(N, 10)
  data = getsamples(X, preps, bases, nshots)
  writesamples(data, path)
  datatest = readsamples(path)
  @test all(data .== datatest)
  writesamples(data, X, path)
  datatest, Xtest = readsamples(path)
  @test all(data .== datatest)

  X = randomstate(N; ξ=2, χ=3)
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
  circuit = randomcircuit(N; depth=depth)
  layer = Tuple[]
  sites = siteinds("Qubit", N)
  outputpath = prefix*"/"*"simulation"
  ϕ = randomstate(sites; χ=10, normalize=true)
  ψ = runcircuit(sites, circuit)
  Ftest = fidelity(ψ, ϕ)
  f(ψ::MPS) = fidelity(ψ, ϕ)#; kwargs...) = fidelity(ψ, ϕ)
  obs = observer(["f" => f])
  ψ = runcircuit(
    sites,
    circuit;
    (observer!)=obs,
    move_sites_back_before_measurements=true,
    outputpath=outputpath,
    savestate=true,
    outputlevel=0,
  )
  @test Ftest ≈ obs[end, "f"]
  @test length(obs[!, "f"]) == depth + 1

  ## # File is out of date
  ## obs2 = load("simulation_observer.jld2")
  ## for (k, v) in obs2
  ##   @test last(v) ≈ obs[!, k]
  ## end

  fin = h5open(prefix*"/"*"simulation_state.h5", "r")
  M = read(fin, "state", MPS)
  close(fin)
  @test M ≈ ψ
end

@testset "circuit observer: MPO" begin
  N = 4
  depth = 5
  #circuit = Vector{Vector{<:Any}}(undef, depth)
  sites = siteinds("Qubit", N)
  circuit = randomcircuit(N; depth=depth)

  @disable_warn_order begin
    L = randomstate(sites; χ=10, ξ=3, normalize=true)
    ϱ = MPO(L)
    ρ = runcircuit(sites, circuit; noise=("DEP", (p=0.001,)))
    Ftest = fidelity(ϱ, ρ)
    g(ρ::MPO; kwargs...) = fidelity(ρ, ϱ)#; kwargs...) = fidelity(ψ, ϕ)
    obs = observer(["g" => g])
    outputpath = prefix*"/"*"simulation"
    ρ₀ = projector(productstate(sites))
    ρ = runcircuit(
      ρ₀,
      circuit;
      (observer!)=obs,
      move_sites_back_before_measurements=true,
      outputlevel=0,
      noise=("DEP", (p=0.001,)),
      outputpath=outputpath,
      savestate=true,
    )
  end
  @test Ftest ≈ obs[end, "g"]
  @test length(obs[!, "g"]) == depth + 1

  ## # File is out of date
  ## obs2 = load("simulation_observer.jld2")
  ## for (k, v) in obs2
  ##   @test last(v) ≈ obs[!, k]
  ## end

  fin = h5open(prefix*"/"*"simulation_state.h5", "r")
  M = read(fin, "state", MPO)
  close(fin)
  @test M ≈ ρ
end

@testset "state tomography observer output: MPS" begin
  Random.seed!(1234)
  N = 4
  depth = 4
  nshots = 100
  circuit = randomcircuit(N; depth=depth)
  Ψ = runcircuit(circuit)
  bases = randombases(N, 2)
  data = getsamples(Ψ, bases, nshots)
  test_data = copy(data[1:10, :])

  N = length(Ψ)     # Number of productstate
  χ = maxlinkdim(Ψ) # Bond dimension of variational MPS
  ψ0 = randomstate(Ψ; χ=χ, σ=0.1)
  opt = Optimisers.Descent(0.01)

  F(ψ::MPS; kwargs...) = fidelity(ψ, Ψ)
  obs = observer(["F" => F])
  epochs = 18

  batchsize = 10
  observe_step = 3

  outputpath = prefix*"/"*"simulation"
  ψ = tomography(
    data,
    ψ0;
    test_data=test_data,
    batchsize=10,
    epochs=epochs,
    (observer!)=obs,
    observe_step=observe_step,
    outputpath=outputpath,
    savestate=true,
    outputlevel=0,
  )
  @test length(obs[!, "F"]) == epochs ÷ observe_step

  ## # File is out of date
  ## obs2 = load("simulation_observer.jld2")
  ## for (k, v) in obs2
  ##   @test last(v) ≈ obs[!, k]
  ## end

  fin = h5open(prefix*"/"*"simulation_state.h5", "r")
  M = read(fin, "state", MPS)
  close(fin)
  @test M ≈ ψ
end

@testset "state tomography observer output: LPDO" begin
  Random.seed!(1234)
  N = 4
  depth = 4
  nshots = 100
  circuit = randomcircuit(N; depth=depth)
  ϱ = runcircuit(circuit; noise=("DEP", (p=0.01,)))
  bases = randombases(N, 2)
  data = getsamples(ϱ, bases, nshots)
  test_data = copy(data[1:10, :])

  N = length(ϱ)     # Number of productstate
  χ = maxlinkdim(ϱ) # Bond dimension of variational MPS
  ρ = randomstate(ϱ; χ=χ ÷ 2, ξ=2)
  opt = Optimisers.Descent(0.01)

  F(ρ::LPDO; kwargs...) = fidelity(ρ, ϱ)
  obs = observer(["F" => F])
  epochs = 9

  batchsize = 10
  observe_step = 3

  outputpath = prefix*"/"*"simulation"
  ρ = tomography(
    data,
    ρ;
    test_data=test_data,
    batchsize=10,
    epochs=epochs,
    (observer!)=obs,
    observe_step=observe_step,
    outputpath=outputpath,
    savestate=true,
    outputlevel=0,
  )
  @test length(obs[!, "F"]) == epochs ÷ observe_step

  ## # File is out of date
  ## obs2 = load("simulation_observer.jld2")
  ## for (k, v) in obs2
  ##   @test last(v) ≈ obs[!, k]
  ## end

  fin = h5open(prefix*"/"*"simulation_state.h5", "r")
  M = read(fin, "state", LPDO{MPO})
  close(fin)
  @test M.X ≈ ρ.X
end

@testset "process tomography observer output: MPO" begin
  Random.seed!(1234)
  N = 3
  depth = 4
  nshots = 100
  circuit = randomcircuit(N; depth=depth)
  V = runcircuit(circuit; process=true)
  preps = randompreparations(N, 2)
  bases = randombases(N, 2)
  data = getsamples(V, preps, bases, nshots)
  test_data = copy(data[1:10, :])
  N = length(V)     # Number of productstate
  χ = maxlinkdim(V) # Bond dimension of variational MPS
  U0 = randomprocess(V; χ=χ, σ=0.1)
  opt = Optimisers.Descent(0.01)

  F(U::MPO; kwargs...) = fidelity(U, V; process=true)
  obs = observer(["F" => F])
  epochs = 9

  batchsize = 10
  observe_step = 3

  outputpath = prefix*"/"*"simulation"
  U = tomography(
    data,
    U0;
    test_data=test_data,
    batchsize=10,
    epochs=epochs,
    (observer!)=obs,
    observe_step=observe_step,
    outputpath=outputpath,
    savestate=true,
    outputlevel=0,
  )

  @test length(obs[!, "F"]) == epochs ÷ observe_step

  ## # File is out of date
  ## obs2 = load("simulation_observer.jld2")
  ## for (k, v) in obs2
  ##   @test last(v) ≈ obs[!, k]
  ## end

  fin = h5open(prefix*"/"*"simulation_state.h5", "r")
  M = read(fin, "state", MPO)
  close(fin)
  @test M ≈ U
end

@testset "state tomography observer output: LPDO" begin
  Random.seed!(1234)
  N = 2
  depth = 4
  nshots = 100
  circuit = randomcircuit(N; depth=depth)
  Φ = runcircuit(circuit; process=true, noise=("DEP", (p=0.01,)))
  preps = randompreparations(N, 2)
  bases = randombases(N, 2)
  data = getsamples(Φ, preps, bases, nshots)
  test_data = copy(data[1:10, :])

  N = length(Φ)     # Number of productstate
  χ = maxlinkdim(Φ) # Bond dimension of variational MPS
  Λ = randomprocess(Φ; χ=χ ÷ 2, ξ=2)
  opt = Optimisers.Descent(0.01)

  F(Λ::LPDO; kwargs...) = fidelity(Λ, Φ)
  obs = observer(["F" => F])
  epochs = 9

  batchsize = 10
  observe_step = 3

  outputpath = prefix*"/"*"simulation"
  Λ = tomography(
    data,
    Λ;
    test_data=test_data,
    batchsize=10,
    epochs=epochs,
    (observer!)=obs,
    observe_step=observe_step,
    outputpath=outputpath,
    savestate=true,
    outputlevel=0,
  )
  @test length(obs[!, "F"]) == epochs ÷ observe_step

  ## # File is out of date
  ## obs2 = load("simulation_observer.jld2")
  ## for (k, v) in obs2
  ##   @test last(v) ≈ obs[!, k]
  ## end

  fin = h5open(prefix*"/"*"simulation_state.h5", "r")
  M = read(fin, "state", LPDO{MPO})
  close(fin)
  @test M.X ≈ Λ.X
end
