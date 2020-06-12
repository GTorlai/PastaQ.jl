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
  gate = quantumgate("I",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [1.,0.,0.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 2
  site_ind = firstind(psi[site],"Site")
  gate = quantumgate("I",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  gate_data = (gate = "I", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
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
  gate = quantumgate("X",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [0.,0.,1.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 2
  site_ind = firstind(psi[site],"Site")
  gate = quantumgate("X",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [0.,1.,0.,0.]

  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 1
  gate_data = (gate = "X", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
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
  gate = quantumgate("Y",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [0.,0.,im,0.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 2
  site_ind = firstind(psi[site],"Site")
  gate = quantumgate("Y",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [0.,im,0.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 1
  gate_data = (gate = "Y", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
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
  gate = quantumgate("Z",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [1.,0.,0.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 2
  site_ind = firstind(psi[site],"Site")
  gate = quantumgate("Z",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [1.,0.,0.,0.]

  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 1
  gate_data = (gate = "Z", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
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
  gate = quantumgate("H",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,0.,1.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 2
  site_ind = firstind(psi[site],"Site")
  gate = quantumgate("H",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,1.,0.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 1
  gate_data = (gate = "H", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
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
  gate_data = (gate = "S", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  @test array(psi[1]) ≈ [1.,0.]
  psi = qubits(1)
  gate_data = (gate = "X", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  gate_data = (gate = "S", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
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
  gate_data = (gate = "T", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  @test array(psi[1]) ≈ [1.,0.]
  psi = qubits(1)
  gate_data = (gate = "X", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  gate_data = (gate = "T", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
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
  gate_data = (gate = "Rx",site=1,params=(θ=θ,))
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  @test array(psi[1]) ≈ [cos(θ/2.),-im*sin(θ/2.)]
  psi = qubits(1)
  applygate!(psi,"X",1)
  gate_data = (gate = "Rx",site=1,params=(θ=θ,))
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
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
  gate_data = (gate = "Ry",site=1,params=(θ=θ,))
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  @test array(psi[1]) ≈ [cos(θ/2.),sin(θ/2.)]
  psi = qubits(1)
  applygate!(psi,"X",1)
  gate_data = (gate = "Ry",site=1,params=(θ=θ,))
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
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
  gate_data = (gate = "Rz",site=1,params=(ϕ=ϕ,))
  gate=makegate(psi,gate_data)
  applygate!(psi,gate)
  @test array(psi[1]) ≈ [exp(-im*ϕ/2.), 0.]
  psi = qubits(1)
  applygate!(psi,"X",1)
  gate_data = (gate = "Rz",site=1,params=(ϕ=ϕ,))
  gate=makegate(psi,gate_data)
  applygate!(psi,gate)
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
  gate_data = (gate = "Rn",site=1,params=(θ=θ,ϕ=ϕ,λ=λ))
  gate=makegate(psi,gate_data)
  applygate!(psi,gate)
  @test array(psi[1]) ≈ [cos(θ/2.),exp(im*ϕ) * sin(θ/2.)]
  psi = qubits(1)
  applygate!(psi,"X",1)
  gate_data = (gate = "Rn",site=1,params=(θ=θ,ϕ=ϕ,λ=λ))
  gate=makegate(psi,gate_data)
  applygate!(psi,gate)
  @test array(psi[1]) ≈ [-exp(im*λ) * sin(θ/2.),exp(im*(ϕ+λ)) * cos(θ/2.)]

end


@testset "apply gate: prep X+" begin
  psi = qubits(1)
  applygate!(psi,"pX+",1)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,1.]
  
  psi = qubits(1)
  gate_data = (gate = "pX+", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,1.]
end


@testset "apply gate: prep X-" begin
  psi = qubits(1)
  applygate!(psi,"pX-",1)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,-1.]
  
  psi = qubits(1)
  gate_data = (gate = "pX-", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,-1.]
end

@testset "apply gate: prep Y+" begin
  psi = qubits(1)
  applygate!(psi,"pY+",1)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,im]
  
  psi = qubits(1)
  gate_data = (gate = "pY+", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,im]
end


@testset "apply gate: prep Y-" begin
  psi = qubits(1)
  applygate!(psi,"pY-",1)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,-im]
  
  psi = qubits(1)
  gate_data = (gate = "pY-", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,-im]
end

@testset "apply gate: prep Z+" begin
  psi = qubits(1)
  applygate!(psi,"pZ+",1)
  @test fullvector(psi) ≈ [1.,0.]
  
  psi = qubits(1)
  gate_data = (gate = "pZ+", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [1.,0.]
end

@testset "apply gate: prep Z-" begin
  psi = qubits(1)
  applygate!(psi,"pZ-",1)
  @test fullvector(psi) ≈ [0.,1.]
  
  psi = qubits(1)
  gate_data = (gate = "pZ-", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [0.,1.]
end

@testset "apply gate: meas X" begin
  psi = qubits(1)
  applygate!(psi,"pX+",1)
  applygate!(psi,"mX",1)
  @test fullvector(psi) ≈ [1.,0.]
  psi = qubits(1)
  applygate!(psi,"pX-",1)
  applygate!(psi,"mX",1)
  @test fullvector(psi) ≈ [0.,1.]

  psi = qubits(1)
  gate_data = (gate = "pX+", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  gate_data = (gate = "mX", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [1.,0.]
  psi = qubits(1)
  gate_data = (gate = "pX-", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  gate_data = (gate = "mX", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [0.,1.]
end

@testset "apply gate: meas Y" begin
  psi = qubits(1)
  applygate!(psi,"pY+",1)
  applygate!(psi,"mY",1)
  @test fullvector(psi) ≈ [1.,0.]
  psi = qubits(1)
  applygate!(psi,"pY-",1)
  applygate!(psi,"mY",1)
  @test fullvector(psi) ≈ [0.,1.]

  psi = qubits(1)
  gate_data = (gate = "pY+", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  gate_data = (gate = "mY", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [1.,0.]
  psi = qubits(1)
  gate_data = (gate = "pY-", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  gate_data = (gate = "mY", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [0.,1.]
end

@testset "apply gate: meas Z" begin
  psi = qubits(1)
  applygate!(psi,"mZ",1)
  @test fullvector(psi) ≈ [1.,0.]
  psi = qubits(1)
  applygate!(psi,"pZ-",1)
  applygate!(psi,"mZ",1)
  @test fullvector(psi) ≈ [0.,1.]

  psi = qubits(1)
  gate_data = (gate = "mZ", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [1.,0.]
  psi = qubits(1)
  gate_data = (gate = "pZ-", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  gate_data = (gate = "mZ", site = 1)
  gate = makegate(psi,gate_data)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [0.,1.]
end



@testset "apply gate: Cx" begin
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Cx",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |11> = (0 0 0 1) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Cx",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
  
  psi = qubits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Cx",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |11> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Cx",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  # TARGET - CONTROL
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Cx",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Cx",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |01> -> |11> = (0 0 0 1) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Cx",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
  
  psi = qubits(2)
  # |11> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Cx",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]

  # USE QUANTUMGATE 
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  site_ind1 = firstind(psi[1],"Site")
  site_ind2 = firstind(psi[2],"Site")
  gate = quantumgate("Cx",site_ind1,site_ind2)
  applygate!(psi,gate)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |11> = (0 0 0 1) (natural order)
  site_ind1 = firstind(psi[1],"Site")
  site_ind2 = firstind(psi[2],"Site")
  gate = quantumgate("X",site_ind1)
  applygate!(psi,gate)
  gate = quantumgate("Cx",site_ind1,site_ind2)
  applygate!(psi,gate)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
  
  psi = qubits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  site_ind1 = firstind(psi[1],"Site")
  site_ind2 = firstind(psi[2],"Site")
  gate = quantumgate("X",site_ind2) 
  applygate!(psi,gate)
  gate = quantumgate("Cx",site_ind1,site_ind2)
  applygate!(psi,gate)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |11> -> |10> = (0 0 1 0) (natural order)
  site_ind1 = firstind(psi[1],"Site")
  site_ind2 = firstind(psi[2],"Site")
  gate = quantumgate("X",site_ind1) 
  applygate!(psi,gate)
  gate = quantumgate("X",site_ind2) 
  applygate!(psi,gate)
  gate = quantumgate("Cx",site_ind1,site_ind2)
  applygate!(psi,gate)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
 
  # TARGET - CONTROL
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  site_ind1 = firstind(psi[2],"Site")
  site_ind2 = firstind(psi[1],"Site")
  gate = quantumgate("Cx",site_ind1,site_ind2)
  applygate!(psi,gate)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  site_ind1 = firstind(psi[2],"Site")
  site_ind2 = firstind(psi[1],"Site")
  gate = quantumgate("X",site_ind2) 
  applygate!(psi,gate)
  gate = quantumgate("Cx",site_ind1,site_ind2)
  applygate!(psi,gate)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |01> -> |11> = (0 0 0 1) (natural order)
  site_ind1 = firstind(psi[2],"Site")
  site_ind2 = firstind(psi[1],"Site")
  gate = quantumgate("X",site_ind1) 
  applygate!(psi,gate)
  gate = quantumgate("Cx",site_ind1,site_ind2)
  applygate!(psi,gate)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
  
  psi = qubits(2)
  # |11> -> |01> = (0 1 0 0) (natural order)
  site_ind1 = firstind(psi[2],"Site")
  site_ind2 = firstind(psi[1],"Site")
  gate = quantumgate("X",site_ind2) 
  applygate!(psi,gate)
  gate = quantumgate("X",site_ind1) 
  applygate!(psi,gate)
  gate = quantumgate("Cx",site_ind1,site_ind2)
  applygate!(psi,gate)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]

  # USE GATE DATA STRUCTURE
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  gate_id = (gate = "Cx",site = [1,2])
  gate = makegate(psi,gate_id)
  applygate!(psi,gate)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |11> = (0 0 0 1) (natural order)
  gate_id = (gate = "X",site=1)
  gate = makegate(psi,gate_id)
  applygate!(psi,gate)
  gate_id = (gate = "Cx",site = [1,2])
  gate = makegate(psi,gate_id)
  applygate!(psi,gate)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
  
  psi = qubits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  gate_id = (gate = "X",site=2)
  gate = makegate(psi,gate_id)
  applygate!(psi,gate)
  gate_id = (gate = "Cx",site = [1,2])
  gate = makegate(psi,gate_id)
  applygate!(psi,gate)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |11> -> |10> = (0 0 1 0) (natural order)
  gate_id = (gate = "X",site=1)
  gate = makegate(psi,gate_id)
  applygate!(psi,gate)
  gate_id = (gate = "X",site=2)
  gate = makegate(psi,gate_id)
  applygate!(psi,gate)
  gate_id = (gate = "Cx",site = [1,2])
  gate = makegate(psi,gate_id)
  applygate!(psi,gate)
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]

end

@testset "apply gate: Cy" begin
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Cy",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> i|11> = (0 0 0 i) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Cy",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,im]
  
  psi = qubits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Cy",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |11> -> -i|10> = (0 0 -i 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Cy",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,-im,0.]
  
  # TARGET - CONTROL
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Cy",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Cy",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |01> -> i|11> = (0 0 0 i) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Cy",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,im]
  
  psi = qubits(2)
  # |11> -> -i|01> = (0 -i 0 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Cy",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,-im,0.,0.]

end

@testset "apply gate: Cz" begin
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Cz",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Cz",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Cz",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |11> -> -|11> = (0 0 0 -1) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Cz",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,-1.]
  

  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Cz",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Cz",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Cz",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |11> -> -|11> = (0 0 0 -1) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Cz",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,-1.]

end

@testset "apply gate: Sw" begin
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Sw",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Sw",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |01> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Sw",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |11> -> |11> = (0 0 0 1) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Sw",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
  
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Sw",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Sw",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |01> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Sw",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |11> -> |11> = (0 0 0 1) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Sw",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]

end

