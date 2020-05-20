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

@testset "circuits: Cx layer" begin
  N = 8
  gates = []
  Cxlayer!(N,gates,"odd")
  @test length(gates) == N ÷ 2
  for j in 1:length(gates)
    @test typeof(gates[j]) == NamedTuple{(:gate, :site),Tuple{String,Tuple{Int64,Int64}}}
    @test gates[j].gate == "Cx"
    @test gates[j].site == (2*j-1,2*j) 
  end
  gates = []
  Cxlayer!(N,gates,"even")
  @test length(gates) == (N ÷ 2) - 1
  for j in 1:length(gates)
    @test typeof(gates[j]) == NamedTuple{(:gate, :site),Tuple{String,Tuple{Int64,Int64}}}
    @test gates[j].gate == "Cx"
    @test gates[j].site == (2*j,2*j+1) 
  end
  
  N = 9
  gates = []
  Cxlayer!(N,gates,"odd")
  @test length(gates) == N ÷ 2
  for j in 1:length(gates)
    @test typeof(gates[j]) == NamedTuple{(:gate, :site),Tuple{String,Tuple{Int64,Int64}}}
    @test gates[j].gate == "Cx"
    @test gates[j].site == (2*j-1,2*j) 
  end
  gates = []
  Cxlayer!(N,gates,"even")
  @test length(gates) == (N ÷ 2) 
  for j in 1:length(gates)
    @test typeof(gates[j]) == NamedTuple{(:gate, :site),Tuple{String,Tuple{Int64,Int64}}}
    @test gates[j].gate == "Cx"
    @test gates[j].site == (2*j,2*j+1) 
  end
end
