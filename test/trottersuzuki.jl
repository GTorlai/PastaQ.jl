using PastaQ
using ITensors
using Test
using Random

@testset "trotter circuit: real dynamics with Tuples" begin
  N = 2
  B = 0.2
  
  sites = qubits(N) 
  
  ampo = OpSum()
  # loop over the pauli operators
  for j in 1:N-1
    ampo += -1.0,"Z",j,"Z",j+1
    ampo += -B,"X",j
  end
  ampo += -B,"X",N
  Hmpo = MPO(ampo, sites)

  #H = Tuple[]
  ## loop over the pauli operators
  #for j in 1:N-1
  #  H = vcat(H, (-1.0, "ZZ", (j,j+1)))
  #  H = vcat(H, (-B, "X", j,))
  #end
  #H = vcat(H, (-B, "X", N,))

  t = 1.0
  ψ₀ = productstate(sites)
  ψtest = noprime(exp(-im * prod(Hmpo) * t) * prod(ψ₀))
  
  # 1. Specify final time and trotter step
  δt = 0.001
  t = 1.0
  circuit = trottercircuit(ampo; δt = δt, t = t)
  @test length(circuit) == 6000
  ψ = runcircuit(sites, circuit)
  @test prod(ψ) ≈ ψtest atol = 1e-5 

  ## 2. Specify  a time sequence 
  #ts = 0:0.001:1.0
  #circuit = trottercircuit(H; ts = ts)
  #@test length(circuit) == 6000
  #ψ = runcircuit(sites, circuit)
  #@test prod(ψ) ≈ ψtest atol = 1e-5 
  #ts = collect(0:0.001:1.0)
  #circuit = trottercircuit(H; ts = ts)
  #@test length(circuit) == 6000
  #ψ = runcircuit(sites, circuit)
  #@test prod(ψ) ≈ ψtest atol = 1e-5 
  #
  ### 3. Specify a non-uniform time-sequence
  ##ts = vcat(collect(0:0.001:0.5), collect(0.502:0.002:1.0))
  ##circuit = trottercircuit(H; ts = ts)
  ###@test length(circuit) == length(ts)-1
  ##ψ = runcircuit(sites, circuit)
  ##@test prod(ψ) ≈ ψtest atol = 1e-5 
end

#@testset "trotter circuit: imaginary dynamics with Tuples" begin
#  N = 2
#  B = 0.2
#  
#  sites = qubits(N) 
# 
#
#  ampo = OpSum()
#  # loop over the pauli operators
#  for j in 1:N-1
#    ampo += -1.0,"Z",j,"Z",j+1
#    ampo += -B,"X",j
#  end
#  ampo += -B,"X",N
#  Hmpo = MPO(ampo, sites)
#
#  H = Tuple[]
#  # loop over the pauli operators
#  for j in 1:N-1
#    H = vcat(H, (-1.0, "ZZ", (j,j+1)))
#    H = vcat(H, (-B, "X", j,))
#  end
#  H = vcat(H, (-B, "X", N,))
#
#  τ = 1.0
#  ψ₀ = productstate(sites)
#  ψtest = noprime(exp(-prod(Hmpo) * τ) * prod(ψ₀))
#  
#  ## 1. Specify final time and trotter step
#  δτ = 0.001
#  circuit = trottercircuit(H; δτ = δτ, τ = τ)
#  @test length(circuit) == 6000
#  ψ = runcircuit(sites, circuit)
#  @test prod(ψ) ≈ ψtest atol = 1e-5 
#
#  # 2. Specify  a time sequence 
#  τs = 0:0.001:1.0
#  circuit = trottercircuit(H; τs = τs)
#  @test length(circuit) == 6000
#  ψ = runcircuit(sites, circuit)
#  @test prod(ψ) ≈ ψtest atol = 1e-5 
#  τs = collect(0:0.001:1.0)
#  circuit = trottercircuit(H; τs = τs)
#  @test length(circuit) == 6000
#  ψ = runcircuit(sites, circuit)
#  @test prod(ψ) ≈ ψtest atol = 1e-5 
#  
#  ## 3. Specify a non-uniform time-sequence
#  #τs = vcat(collect(0:0.001:0.5), collect(0.502:0.002:1.0))
#  #circuit = trottercircuit(H; τs = τs)
#  #@test length(circuit) == length(τs)-1
#  #ψ = runcircuit(sites, circuit)
#  #@test prod(ψ) ≈ ψtest atol = 1e-5 
#end
#
