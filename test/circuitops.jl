using PastaQ
using ITensors
using Test
using LinearAlgebra

@testset "apply gate: Id" begin
  psi = qubits(2)
  gate_data = ("I", 1)
  psi = runcircuit(psi,gate_data)
  @test array(psi) ≈ [1.,0.,0.,0.]
end
  
@testset "apply gate: X" begin
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 1
  gate_data = ("X", 1)
  psi=runcircuit(psi,gate_data)
  @test array(psi) ≈ [0.,0.,1.,0.]
  
end

@testset "apply gate: Y" begin
  psi = qubits(2)
  site = 1
  gate_data = ("Y", 1)
  psi=runcircuit(psi,gate_data)
  @test array(psi) ≈ [0.,0.,im,0.]
  
end

@testset "apply gate: Z" begin
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 1
  gate_data = ("Z", 1)
  psi=runcircuit(psi,gate_data)
  @test array(psi) ≈ [1.,0.,0.,0.]
  
end

@testset "apply gate: H" begin
  # Build gate first, then apply using an ITensor
  psi = qubits(2)
  site = 1
  gate_data = ("H", 1)
  psi=runcircuit(psi,gate_data)
  @test array(psi) ≈ 1/sqrt(2.)*[1.,0.,1.,0.] 
end

@testset "apply gate: S" begin
  psi = qubits(1)
  gate_data = ("S", 1)
  psi=runcircuit(psi,gate_data)
  @test array(psi[1]) ≈ [1.,0.]
  psi = qubits(1)
  gate_data = ("X", 1)
  psi=runcircuit(psi,gate_data)
  gate_data = ("S", 1)
  psi=runcircuit(psi,gate_data)
  @test array(psi[1]) ≈ [0.,im]
end

@testset "apply gate: T" begin
  psi = qubits(1)
  gate_data = ("T", 1)
  psi=runcircuit(psi,gate_data)
  @test array(psi[1]) ≈ [1.,0.]
  psi = qubits(1)
  gate_data = ("X", 1)
  psi=runcircuit(psi,gate_data)
  gate_data = ("T", 1)
  psi=runcircuit(psi,gate_data)
  @test array(psi[1]) ≈ [0.,exp(im*π/4)]
end

@testset "apply gate: Rx" begin
  θ = π * rand()
  psi = qubits(1)
  gate_data = ("Rx",1,(θ=θ,))
  psi=runcircuit(psi,gate_data)
  @test array(psi[1]) ≈ [cos(θ/2.),-im*sin(θ/2.)]
  psi = qubits(1)
  psi=runcircuit(psi,("X",1))
  #PastaQ.applygate!(psi,"X",1)
  gate_data = ("Rx",1,(θ=θ,))
  psi=runcircuit(psi,gate_data)
  @test array(psi[1]) ≈ [-im*sin(θ/2.),cos(θ/2.)]
end

@testset "apply gate: Ry" begin
  θ = π * rand()
  psi = qubits(1)
  gate_data = ("Ry",1,(θ=θ,))
  psi=runcircuit(psi,gate_data)
  @test array(psi[1]) ≈ [cos(θ/2.),sin(θ/2.)]
  psi = qubits(1)
  psi=runcircuit(psi,("X",1))
  gate_data = ("Ry",1,(θ=θ,))
  psi=runcircuit(psi,gate_data)
  @test array(psi[1]) ≈ [-sin(θ/2.),cos(θ/2.)]

end

@testset "apply gate: Rz" begin
  ϕ = 2π * rand()
  psi = qubits(1)
  gate_data = ("Rz",1,(ϕ=ϕ,))
  psi=runcircuit(psi,gate_data)
  @test array(psi[1]) ≈ [1.0, 0.0]
  psi = qubits(1)
  psi=runcircuit(psi,("X",1))
  gate_data = ("Rz",1,(ϕ=ϕ,))
  psi=runcircuit(psi,gate_data)
  @test array(psi[1]) ≈ [0.0, exp(im*ϕ)]
end

@testset "apply gate: Rn" begin
  θ = 1.0
  ϕ = 2.0
  λ = 3.0
  psi = qubits(1)
  gate_data = ("Rn",1,(θ=θ,ϕ=ϕ,λ=λ))
  psi=runcircuit(psi,gate_data)
  @test array(psi[1]) ≈ [cos(θ/2.),exp(im*ϕ) * sin(θ/2.)]
  psi = qubits(1)
  psi=runcircuit(psi,("X",1))
  gate_data = ("Rn",1,(θ=θ,ϕ=ϕ,λ=λ))
  psi=runcircuit(psi,gate_data)
  @test array(psi[1]) ≈ [-exp(im*λ) * sin(θ/2.),exp(im*(ϕ+λ)) * cos(θ/2.)]

end


@testset "apply gate: prep X+/X-" begin
  psi = qubits(1, ["X+"])
  @test array(psi) ≈ 1/sqrt(2.)*[1.,1.]
  
  psi = qubits(1, ["X-"])
  @test array(psi) ≈ 1/sqrt(2.)*[1.,-1.]
end

@testset "apply gate: prep Y+/Y-" begin
  psi = qubits(1, ["Y+"])
  @test array(psi) ≈ 1/sqrt(2.)*[1.,im]
  
  psi = qubits(1, ["Y-"])
  @test array(psi) ≈ 1/sqrt(2.)*[1.,-im]
end

@testset "apply gate: prep Z+/Z-" begin
  psi = qubits(1, ["Z+"])
  @test array(psi) ≈ [1.,0.]
  
  psi = qubits(1, ["Z-"])
  @test array(psi) ≈ [0.,1.]
end

@testset "apply gate: meas X" begin
  psi = qubits(1, ["X+"])
  gate_data = ("basisX", 1, (dag = true,))
  psi=runcircuit(psi,gate_data)
  @test array(psi) ≈ [1.,0.]
  psi = qubits(1, ["X-"])
  gate_data = ("basisX", 1, (dag = true,))
  psi=runcircuit(psi,gate_data)
  @test array(psi) ≈ [0.,1.]
end

@testset "apply gate: meas Y" begin
  psi = qubits(1, ["Y+"])
  gate_data = ("basisY", 1, (dag = true,))
  psi=runcircuit(psi,gate_data)
  @test array(psi) ≈ [1.,0.]
  psi = qubits(1, ["Y-"])
  gate_data = ("basisY", 1, (dag = true,))
  psi=runcircuit(psi,gate_data)
  @test array(psi) ≈ [0.,1.]
end

@testset "apply gate: meas Z" begin
  psi = qubits(1)
  gate_data = ("basisZ", 1, (dag = true,))
  psi=runcircuit(psi,gate_data)
  @test array(psi) ≈ [1.,0.]
  psi = qubits(1, ["Z-"])
  gate_data = ("basisZ", 1, (dag = true,))
  psi=runcircuit(psi,gate_data)
  @test array(psi) ≈ [0.,1.]
end



@testset "apply gate: CX" begin
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  gate_data = ("CX",(1,2))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |11> = (0 0 0 1) (natural order)
  gate_data = ("X",1)
  psi=runcircuit(psi,gate_data)
  gate_data = ("CX",(1,2))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
  
  psi = qubits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  gate_data = ("X",2)
  psi=runcircuit(psi,gate_data)
  gate_data = ("CX",(1,2))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |11> -> |10> = (0 0 1 0) (natural order)
  gate_data = ("X",1)
  psi=runcircuit(psi,gate_data)
  gate_data = ("X",2)
  psi=runcircuit(psi,gate_data)
  gate_data = ("CX",(1,2))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]

end

@testset "apply gate: CY" begin
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  gate_data = ("CY",(1,2))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> i|11> = (0 0 0 i) (natural order)
  gate_data = ("X",1)
  psi=runcircuit(psi,gate_data)
  gate_data = ("CY",(1,2))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,0.,0.,im]
  
  psi = qubits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  gate_data = ("X",2)
  psi=runcircuit(psi,gate_data)
  gate_data = ("CY",(1,2))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |11> -> -i|10> = (0 0 -i 0) (natural order)
  gate_data = ("X",1)
  psi=runcircuit(psi,gate_data)
  gate_data = ("X",2)
  psi=runcircuit(psi,gate_data)
  gate_data = ("CY",(1,2))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,0.,-im,0.]
  
  # TARGET - CONTROL
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  gate_data = ("CY",(2,1))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  gate_data = ("X",1)
  psi=runcircuit(psi,gate_data)
  gate_data = ("CY",(2,1))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |01> -> i|11> = (0 0 0 i) (natural order)
  gate_data = ("X",2)
  psi=runcircuit(psi,gate_data)
  gate_data = ("CY",(2,1))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,0.,0.,im]
  
  psi = qubits(2)
  # |11> -> -i|01> = (0 -i 0 0) (natural order)
  gate_data = ("X",1)
  psi=runcircuit(psi,gate_data)
  gate_data = ("X",2)
  psi=runcircuit(psi,gate_data)
  gate_data = ("CY",(2,1))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,-im,0.,0.]

end

@testset "apply gate: CZ" begin
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  gate_data = ("CZ",(1,2))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  gate_data = ("X",1)
  psi=runcircuit(psi,gate_data)
  gate_data = ("CZ",(1,2))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  gate_data = ("X",2)
  psi=runcircuit(psi,gate_data)
  gate_data = ("CZ",(1,2))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |11> -> -|11> = (0 0 0 -1) (natural order)
  gate_data = ("X",1)
  psi=runcircuit(psi,gate_data)
  gate_data = ("X",2)
  psi=runcircuit(psi,gate_data)
  gate_data = ("CZ",(1,2))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,0.,0.,-1.]
  

  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  gate_data = ("CZ",(2,1))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  gate_data = ("X",1)
  psi=runcircuit(psi,gate_data)
  gate_data = ("CZ",(2,1))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  gate_data = ("X",2)
  psi=runcircuit(psi,gate_data)
  gate_data = ("CZ",(2,1))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |11> -> -|11> = (0 0 0 -1) (natural order)
  gate_data = ("X",1)
  psi=runcircuit(psi,gate_data)
  gate_data = ("X",2)
  psi=runcircuit(psi,gate_data)
  gate_data = ("CZ",(2,1))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,0.,0.,-1.]

end

@testset "apply gate: Sw" begin
  # CONTROL - TARGET
  psi = qubits(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  gate_data = ("Sw",(1,2))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [1.,0.,0.,0.]
  
  psi = qubits(2)
  # |10> -> |01> = (0 1 0 0) (natural order)
  gate_data = ("X",1)
  psi=runcircuit(psi,gate_data)
  gate_data = ("Sw",(1,2))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,1.,0.,0.]
  
  psi = qubits(2)
  # |01> -> |10> = (0 0 1 0) (natural order)
  gate_data = ("X",2)
  psi=runcircuit(psi,gate_data)
  gate_data = ("Sw",(1,2))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,0.,1.,0.]
  
  psi = qubits(2)
  # |11> -> |11> = (0 0 0 1) (natural order)
  gate_data = ("X",1)
  psi=runcircuit(psi,gate_data)
  gate_data = ("X",2)
  psi=runcircuit(psi,gate_data)
  gate_data = ("Sw",(1,2))
  psi=runcircuit(psi,gate_data)
  psi_vec = array(psi)
  @test psi_vec ≈ [0.,0.,0.,1.]
  
end

