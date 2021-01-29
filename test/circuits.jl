using PastaQ
using ITensors
using Test
using Random

@testset "gatelayer" begin
  N = 10
  layer = gatelayer("X",N)
  @test length(layer) == N
  @test all(x -> x == "X", first.(layer))

  qarray = lineararray(N)
  layer = gatelayer("CX", qarray[1])
  @test length(layer) == length(qarray[1])
  @test all(x -> x == "CX", first.(layer))
  layer = gatelayer("CX", qarray[2])
  @test length(layer) == length(qarray[2])
  @test all(x -> x == "CX", first.(layer))
end

@testset "randommparamrs" begin
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
  
  pars = PastaQ.randomparams("Haar",4)
  @test haskey(pars,:random_matrix)
  @test size(pars[:random_matrix]) == (4,4)
  @test pars[:random_matrix] isa Matrix{ComplexF64}
end


@testset "randomlayer" begin

  N = 10
  layer = randomlayer("Rn",N)
  @test length(layer) == N
  @test all(x -> x == "Rn", first.(layer))

  qarray = lineararray(N)
  layer = randomlayer("Haar", qarray[1])
  @test length(layer) == length(qarray[1])
  @test all(x -> x == "Haar", first.(layer))
  layer = randomlayer("Haar", qarray[2])
  @test length(layer) == length(qarray[2])
  @test all(x -> x == "Haar", first.(layer))

end








