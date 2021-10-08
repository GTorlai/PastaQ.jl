using PastaQ
using ITensors
using Test
using Random

@testset "trotter layer" begin
  N = 5
  H = OpSum()
  nterms = 5
  for j in 1:nterms
    h = rand()
    localop = rand(["X","Y","Z"])
    site = rand(1:N)
    H += h, localop, site 
    
    g = rand()
    localop = rand(["CX","CZ"])
    siteA = rand(1:N)
    siteB = rand(deleteat!(collect(1:N),siteA))
    H += g, localop, siteA,siteB
  end

  layer = PastaQ.trotterlayer(H; δt = 0.1)
  twoqubitgates = layer[1:nterms]
  for (j,g) in enumerate(twoqubitgates)
    @test g == layer[end+1-j]
    @test g[2] isa Tuple 
  end
  onequbitgates = layer[nterms+1:2*nterms]
  for g in onequbitgates
    @test g[2] isa Int
  end
  
  cnt = 0
  for g in layer[1:nterms]
    support = g[2]
    if support[1] ≠ cnt && support[2] ≠ cnt
      cnt += 1
    end
    @test support[1] == cnt || support[2] == cnt
  end
end

@testset "trotter circuit: real time" begin

  N = 5
  B = 0.5
  
  ampo = OpSum()
  # loop over the pauli operators
  for j in 1:N-1
    ampo += -1.0,"Z",j,"Z",j+1
    ampo += -B,"X",j
  end
  ampo += -B,"X",N
  
  sites = siteinds("Qubit",N)
  H = MPO(ampo, sites)

  Hop = OpSum()
  # loop over the pauli operators
  for j in 1:N-1
    Hop += -1.0,"ZZ",(j,j+1)
    Hop += -B,"X",j
  end
  Hop += -B,"X",N

  T = 1.0
  
  ψ₀ = productstate(sites)
  ψtest = noprime(exp(-im * prod(H) * T) * prod(ψ₀))

  circuit = trottercircuit(Hop, T; δt = 0.001)
  @test length(circuit) == 1000  
  @test length(circuit[1]) == 13

  circuit = trottercircuit(Hop, T; δt = 0.001, layered = false)
  @test length(circuit) == 13000 
  
  ψ = runcircuit(ψ₀, circuit)
  @test PastaQ.array(ψ) ≈ PastaQ.array(ψtest) atol = 1e-5
end

@testset "trotter circuit: real time" begin

  N = 5
  B = 0.5
  
  ampo = OpSum()
  # loop over the pauli operators
  for j in 1:N-1
    ampo += -1.0,"Z",j,"Z",j+1
    ampo += -B,"X",j
  end
  ampo += -B,"X",N
  
  sites = siteinds("Qubit",N)
  H = MPO(ampo, sites)

  Hop = OpSum()
  # loop over the pauli operators
  for j in 1:N-1
    Hop += -1.0,"ZZ",(j,j+1)
    Hop += -B,"X",j
  end
  Hop += -B,"X",N

  T = 1.0
  
  ψ₀ = productstate(sites)
  ψtest = noprime(exp(-prod(H) * T) * prod(ψ₀))

  circuit = trottercircuit(Hop, T; δτ = 0.001)
  @test length(circuit) == 1000  
  @test length(circuit[1]) == 13

  circuit = trottercircuit(Hop, T; δτ = 0.001, layered = false)
  @test length(circuit) == 13000 
  
  ψ = runcircuit(ψ₀, circuit)
  @test PastaQ.array(ψ) ≈ PastaQ.array(ψtest) atol = 1e-3
end


