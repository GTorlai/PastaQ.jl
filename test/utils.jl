using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random

@testset "fullvector - native" begin
  psi = qubits(1)
  psi_vec = fullvector(psi, reverse = false)
  @test psi_vec ≈ [1., 0.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi, reverse = false)
  @test psi_vec ≈ [0., 1.]

  psi = qubits(2)
  psi_vec = fullvector(psi, reverse = false)
  @test psi_vec ≈ [1., 0., 0., 0.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi, reverse = false)
  @test psi_vec ≈ [0., 1., 0., 0.]
  applygate!(psi,"X",2)
  psi_vec = fullvector(psi, reverse = false)
  @test psi_vec ≈ [0., 0., 0., 1.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi, reverse = false)
  @test psi_vec ≈ [0., 0., 1., 0.]
end

@testset "fullvector - reverse" begin
  psi = qubits(1)
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [1., 0.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0., 1.]

  psi = qubits(2)
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [1., 0., 0., 0.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0., 0., 1., 0.]
  applygate!(psi,"X",2)
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0., 0., 0., 1.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0., 1., 0., 0.]
end

@testset "fullmatrix for Itensor - reverse" begin
  psi = qubits(2)
  # control = 0, target = 0 -> 00 = 1 0 0 0
  applygate!(psi,"CX",(1,2))
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [1, 0, 0, 0]

  psi = qubits(2)
  # control = 0, target = 1 -> 01 = 0 1 0 0
  applygate!(psi,"X",2)
  applygate!(psi,"CX",(1,2))
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0, 1, 0, 0]

  psi = qubits(2)
  # control = 1, target = 0 -> 11 = 0 0 0 1
  applygate!(psi,"X",1)
  applygate!(psi,"CX",(1,2))
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0, 0, 0, 1]

  psi = qubits(2)
  # control = 1, target = 1 -> 10 = 0 0 1 0
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"CX",(1,2))
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0, 0, 1, 0]

  # INVERTED TARGET AND CONTROL
  psi = qubits(2)
  # target = 0, control = 0 -> 00 = 1 0 0 0
  applygate!(psi,"CX",(2,1))
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [1, 0, 0, 0]

  psi = qubits(2)
  # target = 0, control = 1 -> 11 = 0 0 0 1
  applygate!(psi,"X",2)
  applygate!(psi,"CX",(2,1))
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0, 0, 0, 1]

  psi = qubits(2)
  # target = 1, control = 0 -> 10 = 0 0 1 0
  applygate!(psi,"X",1)
  applygate!(psi,"CX",(2,1))
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0, 0, 1, 0]

  psi = qubits(2)
  # target = 1, control = 1 -> 01 = 0 1 0 0
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"CX",(2,1))
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0, 1, 0, 0]
end

@testset "hilbert space" begin
  N = 10
  Ψ = randomstate(N)
  ρ = randomstate(N;mixed=true)
  U = randomstate(N)
  Λ = randomstate(N;mixed=true)

  h = hilbertspace(Ψ)
  @test length(h) == N
  for j in 1:length(N)
    @test hastags(h[j],"Site")
  end
  h = hilbertspace(ρ)
  @test length(h) == N
  for j in 1:length(N)
    @test hastags(h[j],"Site")
  end
  h = hilbertspace(U)
  @test length(h) == N
  for j in 1:length(N)
    @test hastags(h[j],"Site")
  end
  h = hilbertspace(Λ)
  @test length(h) == N
  for j in 1:length(N)
    @test hastags(h[j],"Site")
  end
end

@testset "replace hilbert space tags" begin
  N = 3
  χ = 3
  
  # Check the tags
  # 1. Given a MPS
  Ψ0 = randomstate(N;χ=χ)

  # to MPS
  Ψ = randomstate(N;χ=χ)
  replacehilbertspace!(Ψ,Ψ0)
  for j in 1:length(N)
    @test firstind(Ψ[j],tags="Site") == firstind(Ψ0[j],tags="Site")
  end
  # to MPO
  ρ = randomstate(N;mixed=true,χ=χ)
  replacehilbertspace!(ρ,Ψ0)
  for j in 1:length(N)
    @test firstind(ρ[j],tags="Site",plev=0) == firstind(Ψ0[j],tags="Site")
  end
  # to LPDO
  ρ = randomstate(N;lpdo=true)
  replacehilbertspace!(ρ,Ψ0)
  for j in 1:length(N)
    @test firstind(ρ.X[j],tags="Site",plev=0) == firstind(Ψ0[j],tags="Site")
  end
  #

  # 1. Given a MPO
  ρ0 = randomstate(N;mixed=true,χ=χ)

  # to MPS
  Ψ = randomstate(N;χ=χ)
  replacehilbertspace!(Ψ,ρ0)
  for j in 1:length(N)
    @test firstind(Ψ[j],tags="Site") == firstind(ρ0[j],tags="Site",plev=0)
  end
  
  ## to MPO
  ρ = randomstate(N;mixed=true,χ=χ)
  replacehilbertspace!(ρ,ρ0)
  for j in 1:length(N)
    @test firstind(ρ[j],tags="Site",plev=0) == firstind(ρ0[j],tags="Site",plev=0)
  end
  # to LPDO
  ρ = randomstate(N;lpdo=true,χ=χ)
  replacehilbertspace!(ρ,ρ0)
  for j in 1:length(N)
    @test firstind(ρ.X[j],tags="Site",plev=0) == firstind(ρ0[j],tags="Site",plev=0)
  end

  ## 1. Given a LPDO
  ρ0 = randomstate(N;lpdo=true,χ=χ)

  # to MPS
  Ψ = randomstate(N;χ=χ)
  replacehilbertspace!(Ψ,ρ0)
  for j in 1:length(N)
    @test firstind(Ψ[j],tags="Site") == firstind(ρ0.X[j],tags="Site",plev=0)
  end
  
  ## to MPO
  ρ = randomstate(N;mixed=true,χ=χ)
  replacehilbertspace!(ρ,ρ0)
  for j in 1:length(N)
    @test firstind(ρ[j],tags="Site",plev=0) == firstind(ρ0.X[j],tags="Site",plev=0)
  end
  # to LPDO
  ρ = randomstate(N;lpdo=true,χ=χ)
  replacehilbertspace!(ρ,ρ0)
  for j in 1:length(N)
    @test firstind(ρ.X[j],tags="Site",plev=0) == firstind(ρ0.X[j],tags="Site",plev=0)
  end

end


