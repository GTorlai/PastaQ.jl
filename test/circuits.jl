using PastaQ
using ITensors
using Test
using Random

@testset "circuits: hadamardlayer" begin
  N = 5
  gates = []
  hadamardlayer!(N,gates)
  @test length(gates) == N
  for j in 1:N
    @test typeof(gates[j]) == NamedTuple{(:gate, :site),Tuple{String,Int64}}
    @test gates[j].gate == "H"
    @test gates[j].site == j
  end
end

@testset "circuits: rand1Qrotationlayer" begin
  N = 5
  gates = []
  rand1Qrotationlayer!(N,gates)
  @test length(gates) == N
  for j in 1:N
    @test typeof(gates[j]) == NamedTuple{(:gate, :site, :params),
                                         Tuple{String,Int64,NamedTuple{(:θ, :ϕ, :λ),
                                               Tuple{Float64,Float64,Float64}}}}
    @test gates[j].gate == "Rn"
    @test gates[j].site == j
    @test 0 ≤ gates[j].params.θ ≤ π
    @test 0 ≤ gates[j].params.ϕ ≤ 2π
    @test 0 ≤ gates[j].params.λ ≤ 2π
  end
  
  rng = MersenneTwister(1234)  
  gates = []
  rand1Qrotationlayer!(N,gates,rng=rng)
  @test length(gates) == N
  for j in 1:N
    @test typeof(gates[j]) == NamedTuple{(:gate, :site, :params),
                                         Tuple{String,Int64,NamedTuple{(:θ, :ϕ, :λ),
                                               Tuple{Float64,Float64,Float64}}}}
    @test gates[j].gate == "Rn"
    @test gates[j].site == j
    @test 0 ≤ gates[j].params.θ ≤ π
    @test 0 ≤ gates[j].params.ϕ ≤ 2π
    @test 0 ≤ gates[j].params.λ ≤ 2π
  end
end

