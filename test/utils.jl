using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random

@testset "numberofqubits" begin
  
  @test numberofqubits(("H",2)) == 2
  @test numberofqubits(("CX",(2,5))) == 5

  for i in 1:10
    depth = 4
    N = rand(2:50)
    gates = randomcircuit(N, depth)
    n = numberofqubits(gates)
    @test N == n
  end
end


#@testset "PastaQ.fullvector - native" begin
#  psi = qubits(1)
#  psi_vec = PastaQ.fullvector(psi, reverse = false)
#  @test psi_vec ≈ [1., 0.]
#  PastaQ.applygate!(psi,"X",1)
#  psi_vec = PastaQ.fullvector(psi, reverse = false)
#  @test psi_vec ≈ [0., 1.]
#
#  psi = qubits(2)
#  psi_vec = PastaQ.fullvector(psi, reverse = false)
#  @test psi_vec ≈ [1., 0., 0., 0.]
#  PastaQ.applygate!(psi,"X",1)
#  psi_vec = PastaQ.fullvector(psi, reverse = false)
#  @test psi_vec ≈ [0., 1., 0., 0.]
#  PastaQ.applygate!(psi,"X",2)
#  psi_vec = PastaQ.fullvector(psi, reverse = false)
#  @test psi_vec ≈ [0., 0., 0., 1.]
#  PastaQ.applygate!(psi,"X",1)
#  psi_vec = PastaQ.fullvector(psi, reverse = false)
#  @test psi_vec ≈ [0., 0., 1., 0.]
#end
#
#@testset "PastaQ.fullvector - reverse" begin
#  psi = qubits(1)
#  psi_vec = PastaQ.fullvector(psi, reverse = true)
#  @test psi_vec ≈ [1., 0.]
#  PastaQ.applygate!(psi,"X",1)
#  psi_vec = PastaQ.fullvector(psi, reverse = true)
#  @test psi_vec ≈ [0., 1.]
#
#  psi = qubits(2)
#  psi_vec = PastaQ.fullvector(psi, reverse = true)
#  @test psi_vec ≈ [1., 0., 0., 0.]
#  PastaQ.applygate!(psi,"X",1)
#  psi_vec = PastaQ.fullvector(psi, reverse = true)
#  @test psi_vec ≈ [0., 0., 1., 0.]
#  PastaQ.applygate!(psi,"X",2)
#  psi_vec = PastaQ.fullvector(psi, reverse = true)
#  @test psi_vec ≈ [0., 0., 0., 1.]
#  PastaQ.applygate!(psi,"X",1)
#  psi_vec = PastaQ.fullvector(psi, reverse = true)
#  @test psi_vec ≈ [0., 1., 0., 0.]
#end
#
#@testset "fullmatrix for Itensor - reverse" begin
#  psi = qubits(2)
#  # control = 0, target = 0 -> 00 = 1 0 0 0
#  PastaQ.applygate!(psi,"CX",(1,2))
#  psi_vec = PastaQ.fullvector(psi, reverse = true)
#  @test psi_vec ≈ [1, 0, 0, 0]
#
#  psi = qubits(2)
#  # control = 0, target = 1 -> 01 = 0 1 0 0
#  PastaQ.applygate!(psi,"X",2)
#  PastaQ.applygate!(psi,"CX",(1,2))
#  psi_vec = PastaQ.fullvector(psi, reverse = true)
#  @test psi_vec ≈ [0, 1, 0, 0]
#
#  psi = qubits(2)
#  # control = 1, target = 0 -> 11 = 0 0 0 1
#  PastaQ.applygate!(psi,"X",1)
#  PastaQ.applygate!(psi,"CX",(1,2))
#  psi_vec = PastaQ.fullvector(psi, reverse = true)
#  @test psi_vec ≈ [0, 0, 0, 1]
#
#  psi = qubits(2)
#  # control = 1, target = 1 -> 10 = 0 0 1 0
#  PastaQ.applygate!(psi,"X",1)
#  PastaQ.applygate!(psi,"X",2)
#  PastaQ.applygate!(psi,"CX",(1,2))
#  psi_vec = PastaQ.fullvector(psi, reverse = true)
#  @test psi_vec ≈ [0, 0, 1, 0]
#
#  # INVERTED TARGET AND CONTROL
#  psi = qubits(2)
#  # target = 0, control = 0 -> 00 = 1 0 0 0
#  PastaQ.applygate!(psi,"CX",(2,1))
#  psi_vec = PastaQ.fullvector(psi, reverse = true)
#  @test psi_vec ≈ [1, 0, 0, 0]
#
#  psi = qubits(2)
#  # target = 0, control = 1 -> 11 = 0 0 0 1
#  PastaQ.applygate!(psi,"X",2)
#  PastaQ.applygate!(psi,"CX",(2,1))
#  psi_vec = PastaQ.fullvector(psi, reverse = true)
#  @test psi_vec ≈ [0, 0, 0, 1]
#
#  psi = qubits(2)
#  # target = 1, control = 0 -> 10 = 0 0 1 0
#  PastaQ.applygate!(psi,"X",1)
#  PastaQ.applygate!(psi,"CX",(2,1))
#  psi_vec = PastaQ.fullvector(psi, reverse = true)
#  @test psi_vec ≈ [0, 0, 1, 0]
#
#  psi = qubits(2)
#  # target = 1, control = 1 -> 01 = 0 1 0 0
#  PastaQ.applygate!(psi,"X",1)
#  PastaQ.applygate!(psi,"X",2)
#  PastaQ.applygate!(psi,"CX",(2,1))
#  psi_vec = PastaQ.fullvector(psi, reverse = true)
#  @test psi_vec ≈ [0, 1, 0, 0]
#end



