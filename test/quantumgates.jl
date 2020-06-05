using PastaQ
using ITensors
using HDF5
using JLD
using Test
using LinearAlgebra

@testset "1q gate: Id" begin
  i = Index(2)
  g = gate_I(i) 
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

@testset "1q gate: Rx" begin
  i = Index(2)
  θ = π * rand()
  g = gate_Rx(i,θ=θ) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: Ry" begin
  i = Index(2)
  θ = π * rand()
  g = gate_Ry(i,θ=θ) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: Rz" begin
  i = Index(2)
  ϕ = 2π * rand()
  g = gate_Rz(i,ϕ=ϕ) 
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
  g = gate_Rn(i,θ=θ,ϕ=ϕ,λ=λ)
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: prep X-" begin
  i = Index(2)
  g = prep_Xm(i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: prep X+" begin
  i = Index(2)
  g = prep_Xp(i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: prep Y-" begin
  i = Index(2)
  g = prep_Ym(i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: prep Y+" begin
  i = Index(2)
  g = prep_Yp(i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: prep Z-" begin
  i = Index(2)
  g = prep_Zm(i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: prep Z+" begin
  i = Index(2)
  g = prep_Zp(i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: meas X" begin
  i = Index(2)
  g = meas_X(i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: meas Y" begin
  i = Index(2)
  g = meas_Y(i) 
  @test plev(inds(g)[1]) == 1 
  @test plev(inds(g)[2]) == 0 
  ggdag = g * prime(dag(g),plev=1,1)
  @test ggdag ≈ itensor(Matrix{ComplexF64}(I, 2, 2),inds(ggdag))
end

@testset "1q gate: meas Z" begin
  i = Index(2)
  g = meas_X(i) 
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
  
  g = gate_I(i) 
  x = quantumgate("I",i) 
  @test g ≈ x 
  
  g = gate_X(i) 
  x = quantumgate("X",i) 
  @test g ≈ x 
  
  g = gate_Y(i) 
  x = quantumgate("Y",i) 
  @test g ≈ x 
  
  g = gate_Z(i) 
  x = quantumgate("Z",i) 
  @test g ≈ x 
  
  g = gate_H(i) 
  x = quantumgate("H",i) 
  @test g ≈ x 
  
  g = gate_S(i) 
  x = quantumgate("S",i) 
  @test g ≈ x 
  
  g = gate_T(i) 
  x = quantumgate("T",i) 
  @test g ≈ x 
  
  θ = π * rand()
  g = gate_Rx(i,θ=θ) 
  x = quantumgate("Rx",i,θ=θ)
  @test g ≈ x 

  θ = π * rand()
  g = gate_Ry(i,θ=θ) 
  x = quantumgate("Ry",i,θ=θ)
  @test g ≈ x 
  
  ϕ = 2π * rand()
  g = gate_Rz(i,ϕ=ϕ) 
  x = quantumgate("Rz",i,ϕ=ϕ)
  @test g ≈ x 
  
  angles = rand(3)
  θ = π * angles[1]
  ϕ = 2π * angles[2]
  λ = 2π * angles[3]
  g = gate_Rn(i,θ=θ,ϕ=ϕ,λ=λ)
  x = quantumgate("Rn",i,θ=θ,ϕ=ϕ,λ=λ)
  @test g ≈ x 

  g = prep_Xp(i) 
  x = quantumgate("pX+",i) 
  @test g ≈ x 
  
  g = prep_Xm(i) 
  x = quantumgate("pX-",i) 
  @test g ≈ x 
  
  g = prep_Yp(i) 
  x = quantumgate("pY+",i) 
  @test g ≈ x 
  
  g = prep_Ym(i) 
  x = quantumgate("pY-",i) 
  @test g ≈ x 
  
  g = prep_Zp(i) 
  x = quantumgate("pZ+",i) 
  @test g ≈ x 
  
  g = prep_Zm(i) 
  x = quantumgate("pZ-",i) 
  @test g ≈ x 
  
  g = meas_X(i) 
  x = quantumgate("mX",i) 
  @test g ≈ x 
  
  g = meas_Y(i) 
  x = quantumgate("mY",i) 
  @test g ≈ x 
  
  g = meas_Z(i) 
  x = quantumgate("mZ",i) 
  @test g ≈ x 
  
  g = gate_Sw(i,j)
  x = quantumgate("Sw",i,j)
  @test g ≈ x

  g = gate_Cx(i,j)
  x = quantumgate("Cx",i,j)
  @test g ≈ x
  
  g = gate_Cy(i,j)
  x = quantumgate("Cy",i,j)
  @test g ≈ x
  
  g = gate_Cz(i,j)
  x = quantumgate("Cz",i,j)
  @test g ≈ x

end

