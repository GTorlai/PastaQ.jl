include("../src/PastaQ.jl")
using Main.PastaQ
using HDF5, JLD
using ITensors
using Test
using LinearAlgebra

@testset "1q gate: Id" begin
  i = Index(2)
  g = gate_Id(i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  @test g ≈ itensor(Matrix{Int}(I, 2, 2),inds(g))
end

@testset "1q gate: X" begin
  i = Index(2)
  g = gate_X(i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: Y" begin
  i = Index(2)
  g = gate_Y(i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end


@testset "1q gate: Z" begin
  i = Index(2)
  g = gate_Z(i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: H" begin
  i = Index(2)
  g = gate_H(i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: S" begin
  i = Index(2)
  g = gate_S(i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: T" begin
  i = Index(2)
  g = gate_T(i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: Kp" begin
  i = Index(2)
  g = gate_Kp(i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: Km" begin
  i = Index(2)
  g = gate_Km(i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: Rx" begin
  i = Index(2)
  θ = π * rand()
  g = gate_Rx(i,θ) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: Ry" begin
  i = Index(2)
  θ = π * rand()
  g = gate_Ry(i,θ) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: Rz" begin
  i = Index(2)
  ϕ = 2π * rand()
  g = gate_Rz(i,ϕ) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end
@testset "1q gate: Rn" begin  
  i = Index(2)
  angles = rand(3)
  θ = π * angles[1]
  ϕ = 2π * angles[2]
  λ = 2π * angles[3]
  g = gate_Rn(i,θ,ϕ,λ)
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "2q gate: Sw" begin  
  i = Index(2)
  j = Index(2)
  g = gate_Sw(i,j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1 
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2)),inds(ggdag))
end

@testset "2q gate: Cx" begin  
  i = Index(2)
  j = Index(2)
  g = gate_Cx(i,j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1 
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2)),inds(ggdag))
end

@testset "2q gate: Cy" begin  
  i = Index(2)
  j = Index(2)
  g = gate_Cy(i,j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1 
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2)),inds(ggdag))
end

@testset "2q gate: Cz" begin  
  i = Index(2)
  j = Index(2)
  g = gate_Cz(i,j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1 
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2)),inds(ggdag))
end

@testset "Gate generation" begin
  i = Index(2)
  j = Index(2)
  
  g = gate_Id(i) 
  x = gate("I",i) 
  @test g ≈ x 
  
  g = gate_X(i) 
  x = gate("X",i) 
  @test g ≈ x 
  
  g = gate_Y(i) 
  x = gate("Y",i) 
  @test g ≈ x 
  
  g = gate_Z(i) 
  x = gate("Z",i) 
  @test g ≈ x 
  
  g = gate_H(i) 
  x = gate("H",i) 
  @test g ≈ x 
  
  g = gate_S(i) 
  x = gate("S",i) 
  @test g ≈ x 
  
  g = gate_T(i) 
  x = gate("T",i) 
  @test g ≈ x 
  
  g = gate_Kp(i) 
  x = gate("Kp",i) 
  @test g ≈ x 
  
  g = gate_Km(i) 
  x = gate("Km",i) 
  @test g ≈ x 
  
  θ = π * rand()
  g = gate_Rx(i,θ) 
  x = gate("Rx",i,angles=θ)
  @test g ≈ x 

  θ = π * rand()
  g = gate_Ry(i,θ) 
  x = gate("Ry",i,angles=θ)
  @test g ≈ x 
  
  ϕ = 2π * rand()
  g = gate_Rz(i,ϕ) 
  x = gate("Rz",i,angles=ϕ)
  @test g ≈ x 
  
  angles = rand(3)
  θ = π * angles[1]
  ϕ = 2π * angles[2]
  λ = 2π * angles[3]
  g = gate_Rn(i,θ,ϕ,λ)
  x = gate("Rn",i,angles=[θ,ϕ,λ])
  @test g ≈ x 

  g = gate_Sw(i,j)
  x = gate("Sw",i,j)
  @test g ≈ x

  g = gate_Cx(i,j)
  x = gate("Cx",i,j)
  @test g ≈ x
  
  g = gate_Cy(i,j)
  x = gate("Cy",i,j)
  @test g ≈ x
  
  g = gate_Cz(i,j)
  x = gate("Cz",i,j)
  @test g ≈ x

end

