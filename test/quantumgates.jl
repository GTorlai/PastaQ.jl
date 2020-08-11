using PastaQ
using ITensors
using HDF5
using JLD
using Test
using LinearAlgebra

@testset "Gate generation" begin
  i = Index(2)
  j = Index(2)
  
  g = gate("I",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = gate("X",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = gate("Y",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
 
  g = gate("Z",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = gate("H",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = gate("S",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = gate("T",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
   
  g = gate("prepX+",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = gate("prepX-",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = gate("prepY+",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = gate("prepY-",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = gate("prepZ+",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = gate("prepZ-",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = gate("measX",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = gate("measY",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = gate("measZ",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  θ = π * rand()
  g = gate("Rx",i,θ=θ)
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  θ = π * rand()
  g = gate("Ry",i,θ=θ)
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  ϕ = 2π * rand()
  g = gate("Rz",i,ϕ=ϕ)
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  angles = rand(3)
  θ = π * angles[1]
  ϕ = 2π * angles[2]
  λ = 2π * angles[3]
  g = gate("Rn",i,θ=θ,ϕ=ϕ,λ=λ)
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = gate("SWAP",i,j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1 
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2))

  g = gate("CX",i,j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1 
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2))
  
  g = gate("CY",i,j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1 
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2))
  
  g = gate("CZ",i,j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1 
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2))
 
end



