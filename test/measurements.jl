using PastaQ
using ITensors
using Random
using Test

@testset "measure one-body operators" begin

  N = 10
  depth = 10
  ψ = runcircuit(randomcircuit(N,depth))
  
  @test measure(ψ,("X",2)) ≈ inner(ψ,runcircuit(ψ,("X",2)))  
  @test measure(ψ,("Y",1)) ≈ inner(ψ,runcircuit(ψ,("Y",1))) 
  @test measure(ψ,("Z",4)) ≈ inner(ψ,runcircuit(ψ,("Z",4))) 
  
  results = measure(ψ,"X")
  for j in 1:length(ψ)
    @test results[j] ≈ inner(ψ,runcircuit(ψ,("X",j)))
  end

  results = measure(ψ,("X",1:2:length(ψ)))
  for j in 1:2:length(ψ)
    @test results[(j+1)÷2] ≈ inner(ψ,runcircuit(ψ,("X",j)))
  end

  results = measure(ψ,("Y",[1,3,5]))
  for j in [1,3,5]
    @test results[(j+1)÷2] ≈ inner(ψ,runcircuit(ψ,("Y",j)))
  end
end
