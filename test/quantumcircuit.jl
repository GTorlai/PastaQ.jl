using PastaQ
using ITensors
using HDF5
using JLD
using Test
using LinearAlgebra

@testset "qubits initialization" begin
  N = 1
  psi = initializequbits(N)
  @test length(psi) == 1
  @test length(inds(psi[1],"Link")) == 0
  @test fullvector(psi) ≈ [1, 0]
  N = 5
  psi = initializequbits(N)
  @test length(psi) == 5
  psi_vec = fullvector(psi)
  exact_vec = zeros(1<<N)
  exact_vec[1] = 1.0
  @test psi_vec ≈ exact_vec
end

@testset "circuit initialization" begin
  N=5
  U = initializecircuit(N)
  @test length(U) == 5
  identity = itensor(reshape([1 0;0 1],(1,2,2)),inds(U[1]))
  @test U[1] ≈ identity
  for s in 2:N-1
    identity = itensor(reshape([1 0;0 1],(1,1,2,2)),inds(U[s]))
    @test U[s] ≈ identity
  end
  identity = itensor(reshape([1 0;0 1],(1,2,2)),inds(U[N]))
  @test U[N] ≈ identity
end

@testset "apply gate: Id" begin
  # Apply gate using gate informations
  psi = initializequbits(1)
  applygate!(psi,"I",1)
  @test fullvector(psi) ≈ [1.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = initializequbits(2)
  site = 1
  site_ind = firstind(psi[site],"Site")
  gate = quantumgate("I",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [1.,0.,0.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = initializequbits(2)
  site = 2
  site_ind = firstind(psi[site],"Site")
  gate = quantumgate("I",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [1.,0.,0.,0.]

end
  
@testset "apply gate: X" begin
  # Apply gate using gate informations
  psi = initializequbits(1)
  applygate!(psi,"X",1)
  @test fullvector(psi) ≈ [0.,1.]
  applygate!(psi,"X",1)
  @test fullvector(psi) ≈ [1.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = initializequbits(2)
  site = 1
  site_ind = firstind(psi[site],"Site")
  gate = quantumgate("X",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [0.,0.,1.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = initializequbits(2)
  site = 2
  site_ind = firstind(psi[site],"Site")
  gate = quantumgate("X",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [0.,1.,0.,0.]
end

@testset "apply gate: Y" begin
  # Apply gate using gate informations
  psi = initializequbits(1)
  applygate!(psi,"Y",1)
  @test fullvector(psi) ≈ [0.,im]
  psi = initializequbits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"Y",1)
  @test fullvector(psi) ≈ [-im,0.]
  
  # Build gate first, then apply using an ITensor
  psi = initializequbits(2)
  site = 1
  site_ind = firstind(psi[site],"Site")
  gate = quantumgate("Y",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [0.,0.,im,0.]
  
  # Build gate first, then apply using an ITensor
  psi = initializequbits(2)
  site = 2
  site_ind = firstind(psi[site],"Site")
  gate = quantumgate("Y",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [0.,im,0.,0.]
end

@testset "apply gate: Z" begin
  # Apply gate using gate informations
  psi = initializequbits(1)
  applygate!(psi,"Z",1)
  @test fullvector(psi) ≈ [1.,0.]
  psi = initializequbits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"Z",1)
  @test fullvector(psi) ≈ [0.,-1.]
  
  # Build gate first, then apply using an ITensor
  psi = initializequbits(2)
  site = 1
  site_ind = firstind(psi[site],"Site")
  gate = quantumgate("Z",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [1.,0.,0.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = initializequbits(2)
  site = 2
  site_ind = firstind(psi[site],"Site")
  gate = quantumgate("Z",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ [1.,0.,0.,0.]
end

@testset "apply gate: H" begin
  # Apply gate using gate informations
  psi = initializequbits(1)
  applygate!(psi,"H",1)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,1.]
  psi = initializequbits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"H",1)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,-1.]

  # Build gate first, then apply using an ITensor
  psi = initializequbits(2)
  site = 1
  site_ind = firstind(psi[site],"Site")
  gate = quantumgate("H",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,0.,1.,0.]
  
  # Build gate first, then apply using an ITensor
  psi = initializequbits(2)
  site = 2
  site_ind = firstind(psi[site],"Site")
  gate = quantumgate("H",site_ind)
  applygate!(psi,gate)
  @test fullvector(psi) ≈ 1/sqrt(2.)*[1.,1.,0.,0.]
  
end

@testset "apply gate: S" begin
  psi = initializequbits(1)
  applygate!(psi,"S",1)
  @test array(psi[1]) ≈ [1.,0.]
  psi = initializequbits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"S",1)
  @test array(psi[1]) ≈ [0.,im]
end

@testset "apply gate: T" begin
  psi = initializequbits(1)
  applygate!(psi,"T",1)
  @test array(psi[1]) ≈ [1.,0.]
  psi = initializequbits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"T",1)
  @test array(psi[1]) ≈ [0.,exp(im*π/4)]
end

@testset "apply gate: Kp" begin
  psi = initializequbits(1)
  applygate!(psi,"Kp",1)
  @test array(psi[1]) ≈ 1/sqrt(2.)*[1.,im]
  psi = initializequbits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"Kp",1)
  @test array(psi[1]) ≈ 1/sqrt(2.)*[1.,-im]
end

@testset "apply gate: Km" begin
  psi = initializequbits(1)
  applygate!(psi,"Km",1)
  @test array(psi[1]) ≈ 1/sqrt(2.)*[1.,1.]
  psi = initializequbits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"Km",1)
  @test array(psi[1]) ≈ 1/sqrt(2.)*[-im,im]
end

@testset "apply gate: Rx" begin
  θ = π * rand()
  psi = initializequbits(1)
  applygate!(psi,"Rx",1,θ=θ)
  @test array(psi[1]) ≈ [cos(θ/2.),-im*sin(θ/2.)]
  psi = initializequbits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"Rx",1,θ=θ)
  @test array(psi[1]) ≈ [-im*sin(θ/2.),cos(θ/2.)]
end

@testset "apply gate: Ry" begin
  θ = π * rand()
  psi = initializequbits(1)
  applygate!(psi,"Ry",1,θ=θ)
  @test array(psi[1]) ≈ [cos(θ/2.),sin(θ/2.)]
  psi = initializequbits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"Ry",1,θ=θ)
  @test array(psi[1]) ≈ [-sin(θ/2.),cos(θ/2.)]
end

@testset "apply gate: Rz" begin
  ϕ = 2π * rand()
  psi = initializequbits(1)
  applygate!(psi,"Rz",1,ϕ=ϕ)
  @test array(psi[1]) ≈ [exp(-im*ϕ/2.), 0.]
  psi = initializequbits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"Rz",1,ϕ=ϕ)
  @test array(psi[1]) ≈ [0.,exp(im*ϕ/2.)]
end

@testset "apply gate: Rn" begin
  angles = rand(3)
  θ = π * angles[1]
  ϕ = 2π * angles[2]
  λ = 2π * angles[3]
  psi = initializequbits(1)
  applygate!(psi,"Rn",1,θ=θ,ϕ=ϕ,λ=λ)
  @test array(psi[1]) ≈ [cos(θ/2.),exp(im*ϕ) * sin(θ/2.)]
  psi = initializequbits(1)
  applygate!(psi,"X",1)
  applygate!(psi,"Rn",1,θ=θ,ϕ=ϕ,λ=λ)
  @test array(psi[1]) ≈ [-exp(im*λ) * sin(θ/2.),exp(im*(ϕ+λ)) * cos(θ/2.)]
end

@testset "apply gate: Cx" begin
  # CONTROL - TARGET
  psi = initializequbits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Cx",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = initializequbits(2)
  # |10> -> |11> = (0 0 0 1) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Cx",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
  
  psi = initializequbits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Cx",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = initializequbits(2)
  # |11> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Cx",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  # TARGET - CONTROL
  psi = initializequbits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Cx",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = initializequbits(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Cx",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = initializequbits(2)
  # |01> -> |11> = (0 0 0 1) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Cx",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
  
  psi = initializequbits(2)
  # |11> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Cx",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]

  # USE APPLYGATE ON ITENSOR
  
  ## CONTROL - TARGET
  #psi = initializequbits(2)
  ## |00> -> |00> = (1 0 0 0) (natural order)
  #site_ind1 = firstind(psi[1],"Site")
  #site_ind2 = firstind(psi[2],"Site")
  #gate = quantumgate("Cx",site_ind1,site_ind2)
  #applygate!(psi,gate)
  #psi_vec = fullvector(psi)
  #@test psi_vec ≈ [1.,0.,0.,0.]
  
  ## TARGET - CONTROL
  #psi = initializequbits(2)
  ## |00> -> |00> = (1 0 0 0) (natural order)
  #applygate!(psi,"Cx",[2,1])
  #psi_vec = fullvector(psi)
  #@test psi_vec ≈ [1.,0.,0.,0.]
  

end

@testset "apply gate: Cy" begin
  # CONTROL - TARGET
  psi = initializequbits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Cy",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = initializequbits(2)
  # |10> -> i|11> = (0 0 0 i) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Cy",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,im]
  
  psi = initializequbits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Cy",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = initializequbits(2)
  # |11> -> -i|10> = (0 0 -i 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Cy",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,-im,0.]
  
  # TARGET - CONTROL
  psi = initializequbits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Cy",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = initializequbits(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Cy",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = initializequbits(2)
  # |01> -> i|11> = (0 0 0 i) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Cy",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,im]
  
  psi = initializequbits(2)
  # |11> -> -i|01> = (0 -i 0 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Cy",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,-im,0.,0.]

end

@testset "apply gate: Cz" begin
  # CONTROL - TARGET
  psi = initializequbits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Cz",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = initializequbits(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Cz",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = initializequbits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Cz",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = initializequbits(2)
  # |11> -> -|11> = (0 0 0 -1) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Cz",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,-1.]
  

  # CONTROL - TARGET
  psi = initializequbits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Cz",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = initializequbits(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Cz",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = initializequbits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Cz",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = initializequbits(2)
  # |11> -> -|11> = (0 0 0 -1) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Cz",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,-1.]

end

@testset "apply gate: Sw" begin
  # CONTROL - TARGET
  psi = initializequbits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Sw",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = initializequbits(2)
  # |10> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Sw",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = initializequbits(2)
  # |01> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Sw",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = initializequbits(2)
  # |11> -> |11> = (0 0 0 1) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Sw",[1,2])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
  
  # CONTROL - TARGET
  psi = initializequbits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  applygate!(psi,"Sw",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = initializequbits(2)
  # |10> -> |01> = (0 1 0 0) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"Sw",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = initializequbits(2)
  # |01> -> |10> = (0 0 1 0) (natural order)
  applygate!(psi,"X",2)
  applygate!(psi,"Sw",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = initializequbits(2)
  # |11> -> |11> = (0 0 0 1) (natural order)
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"Sw",[2,1])
  psi_vec = fullvector(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]

end

#data = load("testdata/quantumcircuit_testunitary_rand1Qrotationlayer.jld")
#N = data["N"]
#gates = data["gates"]
#full_unitary = data["full_unitary"]
#
#@show gates
#@show full_unitary





