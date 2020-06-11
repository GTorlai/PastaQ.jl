using PastaQ
using ITensors
using HDF5
using JLD
using Test
using LinearAlgebra

@testset "Gate generation" begin
  i = Index(2)
  j = Index(2)
  
  g = quantumgate("I",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = quantumgate("X",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = quantumgate("Y",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
 
  g = quantumgate("Z",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = quantumgate("H",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = quantumgate("S",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  g = quantumgate("T",i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
   
  θ = π * rand()
  g = quantumgate("Rx",i,θ=θ)
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  θ = π * rand()
  g = quantumgate("Ry",i,θ=θ)
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  ϕ = 2π * rand()
  g = quantumgate("Rz",i,ϕ=ϕ)
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  angles = rand(3)
  θ = π * angles[1]
  ϕ = 2π * angles[2]
  λ = 2π * angles[3]
  g = quantumgate("Rn",i,θ=θ,ϕ=ϕ,λ=λ)
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  
  
  g = quantumgate("Sw",i,j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1 
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2))

  g = quantumgate("Cx",i,j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1 
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2))
  
  g = quantumgate("Cy",i,j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1 
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2))
  
  g = quantumgate("Cz",i,j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1 
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2))
 
end

