using PastaQ
using ITensors
using Test
using Random

@testset "circuits: appendlayer!" begin
  N = 5
  gates = Tuple[]
  appendlayer!(gates, "H", N)
  @test length(gates) == N
  for j in 1:N
    @test gates[j] isa Tuple{String,Int64}
    @test gates[j][1] == "H"
    @test gates[j][2] == j
  end
end

@testset "circuits: rand1Qrotationlayer" begin
  N = 5
  gates = Tuple[]
  randomrotationlayer!(gates,N)
  @test length(gates) == N
  for j in 1:N
    @test typeof(gates[j]) == Tuple{String,Int64,NamedTuple{(:θ, :ϕ, :λ),
                                    Tuple{Float64,Float64,Float64}}}
    @test gates[j][1] == "Rn"
    @test gates[j][2] == j
    @test 0 ≤ gates[j][3].θ ≤ π
    @test 0 ≤ gates[j][3].ϕ ≤ 2π
    @test 0 ≤ gates[j][3].λ ≤ 2π
  end
  
  #rng = MersenneTwister(1234)  
  gates = Tuple[]
  randomrotationlayer!(gates,N)
  @test length(gates) == N
  for j in 1:N
    @test typeof(gates[j]) == Tuple{String,Int64,NamedTuple{(:θ, :ϕ, :λ),
                                    Tuple{Float64,Float64,Float64}}}
    @test gates[j][1] == "Rn"
    @test gates[j][2] == j
    @test 0 ≤ gates[j][3].θ ≤ π
    @test 0 ≤ gates[j][3].ϕ ≤ 2π
    @test 0 ≤ gates[j][3].λ ≤ 2π
  end
end

