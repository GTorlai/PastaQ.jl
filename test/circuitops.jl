using PastaQ
using ITensors
using HDF5
using JLD
using Test
using LinearAlgebra

@testset "apply gate: Id" begin
  # Apply gate using gate informations
  psi = qubits(1)
  applygate!(psi,"I",1)
  @test fullvector(psi) ≈ [1.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 1
  site_ind = firstind(psi[site],"Site")
  g = gate("I",site_ind)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [1.,0.,0.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 2
  site_ind = firstind(psi[site],"Site")
  g = gate("I",site_ind)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  gate_data = ("I", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [1.,0.,0.,0.]
end
  
@testset "apply gate: X" begin
  # Apply gate using gate informations
  psi = qubits(1)
  applygate!(psi,"X",1)
  @test fullvector(psi) ≈ [0.,1.]
  applygate!(psi,"X",1)
  @test fullvector(psi) ≈ [1.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 1
  site_ind = firstind(psi[site],"Site")
  g = gate("X",site_ind)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [0.,0.,1.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 2
  site_ind = firstind(psi[site],"Site")
  g = gate("X",site_ind)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [0.,1.,0.,0.]

  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 1
  gate_data = ("X", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [0.,0.,1.,0.]
  
end

@testset "apply gate: Y" begin
  # Apply gate using gate informations
  psi = qubits(1)
  applygate!(psi,"Y",1)
  @test fullvector(psi) ≈ [0.,im]
  psi = qubits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"Y",1)
  @test fullvector(psi) ≈ [-im,0.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 1
  site_ind = firstind(psi[site],"Site")
  g = gate("Y",site_ind)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [0.,0.,im,0.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 2
  site_ind = firstind(psi[site],"Site")
  g = gate("Y",site_ind)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [0.,im,0.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 1
  gate_data = ("Y", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [0.,0.,im,0.]
  
end

@testset "apply gate: Z" begin
  # Apply gate using gate informations
  psi = qubits(1)
  applygate!(psi,"Z",1)
  @test fullvector(psi) ≈ [1.,0.]
  psi = qubits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"Z",1)
  @test fullvector(psi) ≈ [0.,-1.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 1
  site_ind = firstind(psi[site],"Site")
  g = gate("Z",site_ind)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [1.,0.,0.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 2
  site_ind = firstind(psi[site],"Site")
  g = gate("Z",site_ind)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [1.,0.,0.,0.]

  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 1
  gate_data = ("Z", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [1.,0.,0.,0.]
  
end

@testset "apply gate: H" begin
  # Apply gate using gate informations
  psi = qubits(1)
  applygate!(psi,"H",1)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,1.]
  psi = qubits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"H",1)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,-1.]

  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 1
  site_ind = firstind(psi[site],"Site")
  g = gate("H",site_ind)
  applygate!(psi,g)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,0.,1.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 2
  site_ind = firstind(psi[site],"Site")
  g = gate("H",site_ind)
  applygate!(psi,g)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,1.,0.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 1
  gate_data = ("H", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,0.,1.,0.] 
end

@testset "apply gate: S" begin
  psi = qubits(1)
  applygate!(psi,"S",1)
  @test array(psi[1]) ≈ [1.,0.]
  psi = qubits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"S",1)
  @test array(psi[1]) ≈ [0.,im]

  psi = qubits(1)
  gate_data = ("S", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test array(psi[1]) ≈ [1.,0.]
  psi = qubits(1)
  gate_data = ("X", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  gate_data = ("S", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test array(psi[1]) ≈ [0.,im]
end

@testset "apply gate: T" begin
  psi = qubits(1)
  applygate!(psi,"T",1)
  @test array(psi[1]) ≈ [1.,0.]
  psi = qubits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"T",1)
  @test array(psi[1]) ≈ [0.,exp(im*π/4)]
  
  psi = qubits(1)
  gate_data = ("T", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test array(psi[1]) ≈ [1.,0.]
  psi = qubits(1)
  gate_data = ("X", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  gate_data = ("T", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test array(psi[1]) ≈ [0.,exp(im*π/4)]
end

@testset "apply gate: Rx" begin
  θ = π * rand()
  psi = qubits(1)
  applygate!(psi,"Rx",1,θ=θ)
  @test array(psi[1]) ≈ [cos(θ/2.),-im*sin(θ/2.)]
  psi = qubits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"Rx",1,θ=θ)
  @test array(psi[1]) ≈ [-im*sin(θ/2.),cos(θ/2.)]

  θ = π * rand()
  psi = qubits(1)
  gate_data = ("Rx",1,(θ=θ,))
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test array(psi[1]) ≈ [cos(θ/2.),-im*sin(θ/2.)]
  psi = qubits(1)
  applygate!(psi,"X",1)
  gate_data = ("Rx",1,(θ=θ,))
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test array(psi[1]) ≈ [-im*sin(θ/2.),cos(θ/2.)]
end

@testset "apply gate: Ry" begin
  θ = π * rand()
  psi = qubits(1)
  applygate!(psi,"Ry",1,θ=θ)
  @test array(psi[1]) ≈ [cos(θ/2.),sin(θ/2.)]
  psi = qubits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"Ry",1,θ=θ)
  @test array(psi[1]) ≈ [-sin(θ/2.),cos(θ/2.)]

  θ = π * rand()
  psi = qubits(1)
  gate_data = ("Ry",1,(θ=θ,))
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test array(psi[1]) ≈ [cos(θ/2.),sin(θ/2.)]
  psi = qubits(1)
  applygate!(psi,"X",1)
  gate_data = ("Ry",1,(θ=θ,))
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test array(psi[1]) ≈ [-sin(θ/2.),cos(θ/2.)]

end

@testset "apply gate: Rz" begin
  ϕ = 2π * rand()
  psi = qubits(1)
  applygate!(psi,"Rz",1,ϕ=ϕ)
  @test array(psi[1]) ≈ [exp(-im*ϕ/2.), 0.]
  psi = qubits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"Rz",1,ϕ=ϕ)
  @test array(psi[1]) ≈ [0.,exp(im*ϕ/2.)]
  
  ϕ = 2π * rand()
  psi = qubits(1)
  gate_data = ("Rz",1,(ϕ=ϕ,))
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test array(psi[1]) ≈ [exp(-im*ϕ/2.), 0.]
  psi = qubits(1)
  applygate!(psi,"X",1)
  gate_data = ("Rz",1,(ϕ=ϕ,))
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test array(psi[1]) ≈ [0.,exp(im*ϕ/2.)]
end

@testset "apply gate: Rn" begin
  angles = rand(3)
  θ = π * angles[1]
  ϕ = 2π * angles[2]
  λ = 2π * angles[3]
  psi = qubits(1)
  applygate!(psi,"Rn",1,θ=θ,ϕ=ϕ,λ=λ)
  @test array(psi[1]) ≈ [cos(θ/2.),exp(im*ϕ) * sin(θ/2.)]
  psi = qubits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"Rn",1,θ=θ,ϕ=ϕ,λ=λ)
  @test array(psi[1]) ≈ [-exp(im*λ) * sin(θ/2.),exp(im*(ϕ+λ)) * cos(θ/2.)]

  psi = qubits(1)
  gate_data = ("Rn",1,(θ=θ,ϕ=ϕ,λ=λ))
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test array(psi[1]) ≈ [cos(θ/2.),exp(im*ϕ) * sin(θ/2.)]
  psi = qubits(1)
  applygate!(psi,"X",1)
  gate_data = ("Rn",1,(θ=θ,ϕ=ϕ,λ=λ))
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test array(psi[1]) ≈ [-exp(im*λ) * sin(θ/2.),exp(im*(ϕ+λ)) * cos(θ/2.)]

end


@testset "apply gate: prep X+" begin
  psi = qubits(1)
  applygate!(psi,"prepX+",1)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,1.]
  
  psi = qubits(1)
  gate_data = ("prepX+", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,1.]
end


@testset "apply gate: prep X-" begin
  psi = qubits(1)
  applygate!(psi,"prepX-",1)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,-1.]
  
  psi = qubits(1)
  gate_data = ("prepX-", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,-1.]
end

@testset "apply gate: prep Y+" begin
  psi = qubits(1)
  applygate!(psi,"prepY+",1)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,im]
  
  psi = qubits(1)
  gate_data = ("prepY+", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,im]
end


@testset "apply gate: prep Y-" begin
  psi = qubits(1)
  applygate!(psi,"prepY-",1)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,-im]
  
  psi = qubits(1)
  gate_data = ("prepY-", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,-im]
end

@testset "apply gate: prep Z+" begin
  psi = qubits(1)
  applygate!(psi,"prepZ+",1)
  @test fullvector(psi) ≈ [1.,0.]
  
  psi = qubits(1)
  gate_data = ("prepZ+", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [1.,0.]
end

@testset "apply gate: prep Z-" begin
  psi = qubits(1)
  applygate!(psi,"prepZ-",1)
  @test fullvector(psi) ≈ [0.,1.]
  
  psi = qubits(1)
  gate_data = ("prepZ-", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [0.,1.]
end

@testset "apply gate: meas X" begin
  psi = qubits(1)
  applygate!(psi,"prepX+",1)
  applygate!(psi,"measX",1)
  @test fullvector(psi) ≈ [1.,0.]
  psi = qubits(1)
  applygate!(psi,"prepX-",1)
  applygate!(psi,"measX",1)
  @test fullvector(psi) ≈ [0.,1.]

  psi = qubits(1)
  gate_data = ("prepX+", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  gate_data = ("measX", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [1.,0.]
  psi = qubits(1)
  gate_data = ("prepX-", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  gate_data = ("measX", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [0.,1.]
end

@testset "apply gate: meas Y" begin
  psi = qubits(1)
  applygate!(psi,"prepY+",1)
  applygate!(psi,"measY",1)
  @test fullvector(psi) ≈ [1.,0.]
  psi = qubits(1)
  applygate!(psi,"prepY-",1)
  applygate!(psi,"measY",1)
  @test fullvector(psi) ≈ [0.,1.]

  psi = qubits(1)
  gate_data = ("prepY+", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  gate_data = ("measY", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [1.,0.]
  psi = qubits(1)
  gate_data = ("prepY-", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  gate_data = ("measY", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [0.,1.]
end

@testset "apply gate: meas Z" begin
  psi = qubits(1)
  applygate!(psi,"measZ",1)
  @test fullvector(psi) ≈ [1.,0.]
  psi = qubits(1)
  applygate!(psi,"prepZ-",1)
  applygate!(psi,"measZ",1)
  @test fullvector(psi) ≈ [0.,1.]

  psi = qubits(1)
  gate_data = ("measZ", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [1.,0.]
  psi = qubits(1)
  gate_data = ("prepZ-", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  gate_data = ("measZ", 1)
  g = gate(psi,gate_data)
  applygate!(psi,g)
  @test fullvector(psi) ≈ [0.,1.]
end



@testset "apply gate: CX" begin
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"CX",(1,2))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |11> = (0 0 0 1) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"CX",(1,2))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
  
  psi = qubits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"CX",(1,2))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |11> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"CX",(1,2))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  # TARGET - CONTROL
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"CX",(2,1))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"CX",(2,1))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |01> -> |11> = (0 0 0 1) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"CX",(2,1))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
  
  psi = qubits(2)
  # |11> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"CX",(2,1))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]

  # USE QUANTUMGATE 
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  site_ind1 = firstind(psi[1],"Site")
  site_ind2 = firstind(psi[2],"Site")
  g = gate("CX",site_ind1,site_ind2)
  applygate!(psi,g)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |11> = (0 0 0 1) (natural order)
  site_ind1 = firstind(psi[1],"Site")
  site_ind2 = firstind(psi[2],"Site")
  g = gate("X",site_ind1)
  applygate!(psi,g)
  g = gate("CX",site_ind1,site_ind2)
  applygate!(psi,g)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
  
  psi = qubits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  site_ind1 = firstind(psi[1],"Site")
  site_ind2 = firstind(psi[2],"Site")
  g = gate("X",site_ind2) 
  applygate!(psi,g)
  g = gate("CX",site_ind1,site_ind2)
  applygate!(psi,g)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |11> -> |10> = (0 0 1 0) (natural order)
  site_ind1 = firstind(psi[1],"Site")
  site_ind2 = firstind(psi[2],"Site")
  g = gate("X",site_ind1) 
  applygate!(psi,g)
  g = gate("X",site_ind2) 
  applygate!(psi,g)
  g = gate("CX",site_ind1,site_ind2)
  applygate!(psi,g)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
 
  # TARGET - CONTROL
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  site_ind1 = firstind(psi[2],"Site")
  site_ind2 = firstind(psi[1],"Site")
  g = gate("CX",site_ind1,site_ind2)
  applygate!(psi,g)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  site_ind1 = firstind(psi[2],"Site")
  site_ind2 = firstind(psi[1],"Site")
  g = gate("X",site_ind2)
  applygate!(psi,g)
  g = gate("CX",site_ind1,site_ind2)
  applygate!(psi,g)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |01> -> |11> = (0 0 0 1) (natural order)
  site_ind1 = firstind(psi[2],"Site")
  site_ind2 = firstind(psi[1],"Site")
  g = gate("X",site_ind1) 
  applygate!(psi,g)
  g = gate("CX",site_ind1,site_ind2)
  applygate!(psi,g)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
  
  psi = qubits(2)
  # |11> -> |01> = (0 1 0 0) (natural order)
  site_ind1 = firstind(psi[2],"Site")
  site_ind2 = firstind(psi[1],"Site")
  g = gate("X",site_ind2) 
  applygate!(psi,g)
  g = gate("X",site_ind1) 
  applygate!(psi,g)
  g = gate("CX",site_ind1,site_ind2)
  applygate!(psi,g)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]

  # USE GATE DATA STRUCTURE
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  gate_id = ("CX",(1,2))
  g = gate(psi,gate_id)
  applygate!(psi,g)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |11> = (0 0 0 1) (natural order)
  gate_id = ("X",1)
  g = gate(psi,gate_id)
  applygate!(psi,g)
  gate_id = ("CX",(1,2))
  g = gate(psi,gate_id)
  applygate!(psi,g)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
  
  psi = qubits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  gate_id = ("X",2)
  g = gate(psi,gate_id)
  applygate!(psi,g)
  gate_id = ("CX",(1,2))
  g = gate(psi,gate_id)
  applygate!(psi,g)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |11> -> |10> = (0 0 1 0) (natural order)
  gate_id = ("X",1)
  g = gate(psi,gate_id)
  applygate!(psi,g)
  gate_id = ("X",2)
  g = gate(psi,gate_id)
  applygate!(psi,g)
  gate_id = ("CX",(1,2))
  g = gate(psi,gate_id)
  applygate!(psi,g)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]

end

@testset "apply gate: CY" begin
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"CY",(1,2))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> i|11> = (0 0 0 i) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"CY",(1,2))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,im]
  
  psi = qubits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"CY",(1,2))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |11> -> -i|10> = (0 0 -i 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"CY",(1,2))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,-im,0.]
  
  # TARGET - CONTROL
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"CY",(2,1))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"CY",(2,1))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |01> -> i|11> = (0 0 0 i) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"CY",(2,1))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,im]
  
  psi = qubits(2)
  # |11> -> -i|01> = (0 -i 0 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"CY",(2,1))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,-im,0.,0.]

end

@testset "apply gate: CZ" begin
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"CZ",(1,2))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"CZ",(1,2))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"CZ",(1,2))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |11> -> -|11> = (0 0 0 -1) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"CZ",(1,2))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,-1.]
  

  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"CZ",(2,1))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"CZ",(2,1))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"CZ",(2,1))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |11> -> -|11> = (0 0 0 -1) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"CZ",(2,1))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,-1.]

end

@testset "apply gate: Sw" begin
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Sw",(1,2))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Sw",(1,2))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |01> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Sw",(1,2))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |11> -> |11> = (0 0 0 1) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Sw",(1,2))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
  
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Sw",(2,1))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Sw",(2,1))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |01> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Sw",(2,1))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |11> -> |11> = (0 0 0 1) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Sw",(2,1))
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
end

