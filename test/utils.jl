using PastaQ
using ITensors
using HDF5
using JLD
using Test
using LinearAlgebra

@testset "fullvector - native" begin
  psi = initializequbits(1)
  psi_vec = fullvector(psi,order="native")
  @test psi_vec ≈ [1., 0.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi,order="native")
  @test psi_vec ≈ [0., 1.]

  psi = initializequbits(2)
  psi_vec = fullvector(psi,order="native")
  @test psi_vec ≈ [1., 0., 0., 0.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi,order="native")
  @test psi_vec ≈ [0., 1., 0., 0.]
  applygate!(psi,"X",2)
  psi_vec = fullvector(psi,order="native")
  @test psi_vec ≈ [0., 0., 0., 1.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi,order="native")
  @test psi_vec ≈ [0., 0., 1., 0.]
end

@testset "fullvector - natural" begin
  psi = initializequbits(1)
  psi_vec = fullvector(psi,order="natural")
  @test psi_vec ≈ [1., 0.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi,order="natural")
  @test psi_vec ≈ [0., 1.]

  psi = initializequbits(2)
  psi_vec = fullvector(psi,order="natural")
  @test psi_vec ≈ [1., 0., 0., 0.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi,order="natural")
  @test psi_vec ≈ [0., 0., 1., 0.]
  applygate!(psi,"X",2)
  psi_vec = fullvector(psi,order="natural")
  @test psi_vec ≈ [0., 0., 0., 1.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi,order="natural")
  @test psi_vec ≈ [0., 1., 0., 0.]
end

@testset "fullmatrix for Itensor - natural" begin
  i = Index(2)
  gate = quantumgate("X",i)
  gate_mat = fullmatrix(gate,order="natural")
  @test gate_mat ≈ [0 1;1 0] 
  
  j = Index(2)
  gate = quantumgate("Cx",i,j)

  psi = initializequbits(2)
  # control = 0, target = 0 -> 00 = 1 0 0 0
  applygate!(psi,"Cx",[1,2])
  psi_vec = fullvector(psi,order="natural")
  @test psi_vec ≈ [1, 0, 0, 0]

  psi = initializequbits(2)
  # control = 0, target = 1 -> 01 = 0 1 0 0
  applygate!(psi,"X",2)
  applygate!(psi,"Cx",[1,2])
  psi_vec = fullvector(psi,order="natural")
  @test psi_vec ≈ [0, 1, 0, 0]

  psi = initializequbits(2)
  # control = 1, target = 0 -> 11 = 0 0 0 1
  applygate!(psi,"X",1)
  applygate!(psi,"Cx",[1,2])
  psi_vec = fullvector(psi,order="natural")
  @test psi_vec ≈ [0, 0, 0, 1]

  psi = initializequbits(2)
  # control = 1, target = 1 -> 10 = 0 0 1 0
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Cx",[1,2])
  psi_vec = fullvector(psi,order="natural")
  @test psi_vec ≈ [0, 0, 1, 0]

  # INVERTED TARGET AND CONTROL
  psi = initializequbits(2)
  # target = 0, control = 0 -> 00 = 1 0 0 0
  applygate!(psi,"Cx",[2,1])
  psi_vec = fullvector(psi,order="natural")
  @test psi_vec ≈ [1, 0, 0, 0]

  psi = initializequbits(2)
  # target = 0, control = 1 -> 11 = 0 0 0 1
  applygate!(psi,"X",2)
  applygate!(psi,"Cx",[2,1])
  psi_vec = fullvector(psi,order="natural")
  @test psi_vec ≈ [0, 0, 0, 1]

  psi = initializequbits(2)
  # target = 1, control = 0 -> 10 = 0 0 1 0
  applygate!(psi,"X",1)
  applygate!(psi,"Cx",[2,1])
  psi_vec = fullvector(psi,order="natural")
  @test psi_vec ≈ [0, 0, 1, 0]

  psi = initializequbits(2)
  # target = 1, control = 1 -> 01 = 0 1 0 0
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Cx",[2,1])
  psi_vec = fullvector(psi,order="natural")
  @test psi_vec ≈ [0, 1, 0, 0]

end
