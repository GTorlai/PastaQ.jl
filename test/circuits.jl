using PastaQ
using ITensors
using Test
using Random

@testset "pre-defined ciruits" begin
  N = 10
  @test length(qft(N)) == reduce(+,1:N-1) + N
  @test length(qft(N; inverse = true)) == reduce(+,1:N-1)+ N
  circuit = ghz(N)
  @test length(circuit) == N
  
end

@testset "gatelayer" begin
  N = 10
  layer = gatelayer("X",N)
  @test length(layer) == N
  @test all(x -> x == "X", first.(layer))

  qarray = PastaQ.lineararray(N)
  layer = gatelayer("CX", qarray[1])
  @test length(layer) == length(qarray[1])
  @test all(x -> x == "CX", first.(layer))
  layer = gatelayer("CX", qarray[2])
  @test length(layer) == length(qarray[2])
  @test all(x -> x == "CX", first.(layer))
end

@testset "randommparams" begin
  N = 10
  pars = PastaQ.randomparams("Rx")
  @test haskey(pars,:θ)
  pars = PastaQ.randomparams("Ry")
  @test haskey(pars,:θ)
  pars = PastaQ.randomparams("Rz")
  @test haskey(pars,:ϕ)
  pars = PastaQ.randomparams("Rn")
  @test haskey(pars,:θ)
  @test haskey(pars,:ϕ)
  @test haskey(pars,:λ)
  
  pars = PastaQ.randomparams("HaarRandomUnitary",4)
  @test haskey(pars,:random_matrix)
  @test size(pars[:random_matrix]) == (4,4)
  @test pars[:random_matrix] isa Matrix{ComplexF64}
end


@testset "randomlayer" begin

  N = 10
  layer = randomlayer("Rn",N)
  @test length(layer) == N
  @test all(x -> x == "Rn", first.(layer))

  qarray = PastaQ.lineararray(N)
  layer = randomlayer("HaarRandomUnitary", qarray[1])
  @test length(layer) == length(qarray[1])
  @test all(x -> x == "HaarRandomUnitary", first.(layer))
  layer = randomlayer("HaarRandomUnitary", qarray[2])
  @test length(layer) == length(qarray[2])
  @test all(x -> x == "HaarRandomUnitary", first.(layer))

end


@testset "random circuits" begin
  N = 30
  depth = 10
  circuit = randomcircuit(N,depth; twoqubitgates = "HaarRandomUnitary")
  @test length(circuit) == depth
  for d in 1:depth
    @test all(x->x == "HaarRandomUnitary",first.(circuit[depth]))
  end

  circuit = randomcircuit(N,depth; twoqubitgates = "CX")
  @test size(circuit,1) == depth
  for d in 1:depth
    @test all(x->x == "CX",first.(circuit[depth]))
  end

  circuit = randomcircuit(N,depth; twoqubitgates = "CX", onequbitgates = "Rn")
  @test size(circuit,1) == depth
  
  circuit = randomcircuit(N, depth; twoqubitgates = "HaarRandomUnitary", onequbitgates = ["Rn","X"])
  @test size(circuit,1) == depth
end

