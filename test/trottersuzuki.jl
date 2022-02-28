using PastaQ
using ITensors
using Test
using Random

@testset "trotter circuit: real dynamics" begin
  N = 2
  B = 0.2

  sites = qubits(N)

  ampo = OpSum()
  # loop over the pauli operators
  for j in 1:(N - 1)
    ampo += -1.0, "Z", j, "Z", j + 1
    ampo += -B, "X", j
  end
  ampo += -B, "X", N
  Hmpo = MPO(ampo, sites)

  H = OpSum()
  # loop over the pauli operators
  for j in 1:(N - 1)
    H += -1.0, "ZZ", (j, j + 1)
    H += -B, "X", j
  end
  H += -B, "X", N

  t = 1.0
  ψ₀ = productstate(sites)
  ψtest = noprime(exp(-im * prod(Hmpo) * t) * prod(ψ₀))

  # 1. Specify final time and trotter step
  δt = 0.001
  t = 1.0
  circuit = trottercircuit(H; δt=δt, t=t)
  @test length(circuit) == 1000
  ψ = runcircuit(sites, circuit)
  @test prod(ψ) ≈ ψtest atol = 1e-5

  # 2. Specify  a time sequence 
  ts = 0:0.001:1.0
  circuit = trottercircuit(H; ts=ts)
  @test length(circuit) == 1000
  ψ = runcircuit(sites, circuit)
  @test prod(ψ) ≈ ψtest atol = 1e-5
  ts = collect(0:0.001:1.0)
  circuit = trottercircuit(H; ts=ts)
  @test length(circuit) == 1000
  ψ = runcircuit(sites, circuit)
  @test prod(ψ) ≈ ψtest atol = 1e-5

  # 3. Specify a non-uniform time-sequence
  ts = vcat(collect(0:0.001:0.5), collect(0.502:0.002:1.0))
  circuit = trottercircuit(H; ts=ts)
  @test length(circuit) == length(ts) - 1
  ψ = runcircuit(sites, circuit)
  @test prod(ψ) ≈ ψtest atol = 1e-5
end

@testset "trotter circuit: imaginary dynamics" begin
  N = 2
  B = 0.2

  sites = qubits(N)

  ampo = OpSum()
  # loop over the pauli operators
  for j in 1:(N - 1)
    ampo += -1.0, "Z", j, "Z", j + 1
    ampo += -B, "X", j
  end
  ampo += -B, "X", N
  Hmpo = MPO(ampo, sites)

  H = OpSum()
  # loop over the pauli operators
  for j in 1:(N - 1)
    H += -1.0, "ZZ", (j, j + 1)
    H += -B, "X", j
  end
  H += -B, "X", N

  τ = 1.0
  ψ₀ = productstate(sites)
  ψtest = noprime(exp(-prod(Hmpo) * τ) * prod(ψ₀))

  ## 1. Specify final time and trotter step
  δτ = 0.001
  circuit = trottercircuit(H; δτ=δτ, τ=τ)
  @test length(circuit) == 1000
  ψ = runcircuit(sites, circuit)
  @test prod(ψ) ≈ ψtest atol = 1e-5

  # 2. Specify  a time sequence 
  τs = 0:0.001:1.0
  circuit = trottercircuit(H; τs=τs)
  @test length(circuit) == 1000
  ψ = runcircuit(sites, circuit)
  @test prod(ψ) ≈ ψtest atol = 1e-5
  τs = collect(0:0.001:1.0)
  circuit = trottercircuit(H; τs=τs)
  @test length(circuit) == 1000
  ψ = runcircuit(sites, circuit)
  @test prod(ψ) ≈ ψtest atol = 1e-5

  # 3. Specify a non-uniform time-sequence
  τs = vcat(collect(0:0.001:0.5), collect(0.502:0.002:1.0))
  circuit = trottercircuit(H; τs=τs)
  @test length(circuit) == length(τs) - 1
  ψ = runcircuit(sites, circuit)
  @test prod(ψ) ≈ ψtest atol = 1e-5
end

#@testset "trotter layer 1" begin
#  Random.seed!(1234)
#  N = 20
#  H = OpSum()
#  nterms = 30
#  for j in 1:nterms
#    h = rand()
#    localop = rand(["X","Y","Z"])
#    site = rand(1:N)
#    H += h, localop, site 
#    
#    g = rand()
#    localop = rand(["CX","CZ"])
#    siteA = rand(1:N)
#    siteB = rand(deleteat!(collect(1:N),siteA))
#    H += g, localop, siteA,siteB
#  end
#
#  layer = PastaQ.trotterlayer(H, 0.1)
#  twoqubitgates = layer[1:nterms]
#  @test length(layer) == 2*nterms 
#  @test all([g[2] isa Tuple for g in twoqubitgates]) 
#  onequbitgates = layer[nterms+1:2*nterms]
#  @test all([g[2] isa Int for g in onequbitgates]) 
#end
#
#@testset "trotter layer 2" begin
#  Random.seed!(1234)
#  N = 20
#  H = OpSum()
#  nterms = 30
#  for j in 1:nterms
#    h = rand()
#    localop = rand(["X","Y","Z"])
#    site = rand(1:N)
#    H += h, localop, site 
#    
#    g = rand()
#    localop = rand(["CX","CZ"])
#    siteA = rand(1:N)
#    siteB = rand(deleteat!(collect(1:N),siteA))
#    H += g, localop, siteA,siteB
#  end
#  layer = PastaQ.trotterlayer(H; δt = 0.1)
#  twoqubitgates = layer[1:nterms]
#  for (j,g) in enumerate(twoqubitgates)
#    @test g == layer[end+1-j]
#    @test g[2] isa Tuple 
#  end
#  onequbitgates = layer[nterms+1:3*nterms]
#  for g in onequbitgates
#    @test g[2] isa Int
#  end
#end
