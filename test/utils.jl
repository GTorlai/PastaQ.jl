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
  
  # MPS <- MPS
  Ψ = randomstate(N)
  Φ = randomstate(N)
  Φprod = prod(Φ)
  replacehilbertspace!(Φ,Ψ)
  @test array(Φprod) ≈ array(prod(Φ)) 
  for j in 1:length(N)
    @test firstind(Ψ[j],tags="Site") == firstind(Φ[j],tags="Site")
  end
  
  # MPS <- regular MPO
  Ψ = randomstate(N)
  ρ = randomstate(N;mixed=true)
  ρprod = prod(ρ) 
  replacehilbertspace!(Ψ,ρ)
  @test array(ρprod) ≈ array(prod(ρ))
  #@show ρ0
  #@show Ψ
  for j in 1:length(N)
    @test firstind(Ψ[j],tags="Site") == firstind(ρ[j],tags="Site",plev=0)
  end
  
  # MPS <- purified MPO
  Ψ = randomstate(N)
  ρ = randomstate(N;lpdo=true)
  ρ2 = copy(ρ)
  replacehilbertspace!(Ψ,ρ)
  #@show ρ0
  #@show Ψ
  for j in 1:length(N)
    @test array(ρ.X[j]) ≈ array(ρ2.X[j])
    @test firstind(Ψ[j],tags="Site") == firstind(ρ.X[j],tags="Site",plev=0)
  end

  
  # regular MPO <- MPS
  ρ = randomstate(N;mixed=true)
  ψ = randomstate(N)
  ρ2 = copy(ρ)
  #@show ρ
  #@show ψ
  replacehilbertspace!(ρ,ψ)
  for j in 1:length(N)
    @test array(ρ[j]) ≈ array(ρ2[j])
    @test firstind(ψ[j],tags="Site") == firstind(ρ[j],tags="Site",plev=0)
    @test firstind(ψ[j],tags="Site")' == firstind(ρ[j],tags="Site",plev=1)
  end
  
  # regular MPO <- regular MPO
  ρ = randomstate(N;mixed=true)
  ρ0 = randomstate(N;mixed=true)
  ρ2 = copy(ρ)
  replacehilbertspace!(ρ,ρ0)
  for j in 1:length(N)
    @test array(ρ[j]) ≈ array(ρ2[j])
    @test firstind(ρ0[j],tags="Site",plev=0) == firstind(ρ[j],tags="Site",plev=0)
    @test firstind(ρ0[j],tags="Site",plev=1) == firstind(ρ[j],tags="Site",plev=1)
  end

  # regular MPO <- purified MPO
  ρ = randomstate(N;mixed=true)
  ρ0 = randomstate(N;lpdo=true)
  ρ2 = copy(ρ)
  replacehilbertspace!(ρ,ρ0)
  #@show ρ0
  #@show ρ
  for j in 1:length(N)
    @test array(ρ[j]) ≈ array(ρ2[j])
    @test firstind(ρ0.X[j],tags="Site",plev=0) == firstind(ρ[j],tags="Site",plev=0)
    @test firstind(ρ0.X[j],tags="Site",plev=0)' == firstind(ρ[j],tags="Site",plev=1)
  end


  # purified MPO <- MPS
  ρ = randomstate(N;lpdo=true)
  ψ = randomstate(N)
  #@show ρ
  #@show ψ
  replacehilbertspace!(ρ,ψ)
  for j in 1:length(N)
    @test firstind(ψ[j],tags="Site") == firstind(ρ.X[j],tags="Site",plev=0)
  end
  
  # purified MPO <- regular MPO
  ρ = randomstate(N;lpdo=true)
  ρ0 = randomstate(N;mixed=true)
  #@show ρ
  #@show ρ0
  replacehilbertspace!(ρ,ρ0)
  for j in 1:length(N)
    @test firstind(ρ0[j],tags="Site",plev=0) == firstind(ρ.X[j],tags="Site",plev=0)
    @test firstind(ρ0[j],tags="Site",plev=1) == firstind(ρ.X[j],tags="Site",plev=0)'
  end

  # purified MPO <- purified MPO
  ρ = randomstate(N;lpdo=true)
  ρ0 = randomstate(N;lpdo=true)
  replacehilbertspace!(ρ,ρ0)
  for j in 1:length(N)
    @test firstind(ρ0.X[j],tags="Site",plev=0) == firstind(ρ.X[j],tags="Site",plev=0)
  end
  
  #@show ρ
  #@show ρ0



  # PROCESS
  # regular MPO <- MPS
  ρ = randomprocess(N;mixed=false)
  ψ = randomstate(N)
  replacehilbertspace!(ρ,ψ)
  for j in 1:length(N)
    @test firstind(ψ[j],tags="Site") == firstind(ρ[j],tags="Site",plev=0)
    @test firstind(ψ[j],tags="Site")' == firstind(ρ[j],tags="Site",plev=1)
  end
  
  # regular MPO <- regular MPO
  ρ = randomprocess(N;mixed=false)
  ρ0 = randomstate(N;mixed=true)
  replacehilbertspace!(ρ,ρ0)
  for j in 1:length(N)
    @test firstind(ρ0[j],tags="Site",plev=0) == firstind(ρ[j],tags="Site",plev=0)
    @test firstind(ρ0[j],tags="Site",plev=1) == firstind(ρ[j],tags="Site",plev=1)
  end

  # regular MPO <- purified MPO
  ρ = randomprocess(N;mixed=false)
  ρ0 = randomstate(N;lpdo=true)
  replacehilbertspace!(ρ,ρ0)
  #@show ρ0
  #@show ρ
  for j in 1:length(N)
    @test firstind(ρ0.X[j],tags="Site",plev=0) == firstind(ρ[j],tags="Site",plev=0)
    @test firstind(ρ0.X[j],tags="Site",plev=0)' == firstind(ρ[j],tags="Site",plev=1)
  end

  # purified MPO <- MPS
  ρ = randomprocess(N;mixed=true)
  ψ = randomstate(N)
  replacehilbertspace!(ρ,ψ)
  #@show ρ
  #@show ψ
  for j in 1:length(N)
    @test firstind(ψ[j],tags="Site") == firstind(ρ.X[j],tags="Site",plev=0)
  end
  
  # purified MPO <- MPS
  ρ = randomprocess(N;mixed=true)
  ψ = randomstate(N)
  replacehilbertspace!(ρ,ψ)
  #@show ρ
  #@show ψ
  for j in 1:length(N)
    @test firstind(ψ[j],tags="Site") == firstind(ρ.X[j],tags="Site",plev=0)
  end

end


