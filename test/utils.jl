using PastaQ
using ITensors
using Test
using LinearAlgebra

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
