include("../src/PastaQ.jl")
using Main.PastaQ
using HDF5, JLD
using ITensors
using Test
using LinearAlgebra

@testset "One-qubit gates" begin
  gates = QuantumGates()
  
  # Test X gate
  gg_dag = gates.X * dag(prime(gates.X,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity
  
  # Test Y gate
  gg_dag = gates.Y * dag(prime(gates.Y,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity 
  
  # Test Z gate
  gg_dag = gates.Z * dag(prime(gates.Z,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity
  
  # Test H gate
  gg_dag = gates.H * dag(prime(gates.H,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity
  
  # Test S gate
  gg_dag = gates.S * dag(prime(gates.S,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity
  
  # Test conjugate S gate
  gg_dag = gates.Sdg * dag(prime(gates.Sdg,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity
  
  # Test T gate
  gg_dag = gates.T * dag(prime(gates.T,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity
  
  # Test Kp gate
  gg_dag = gates.Kp * dag(prime(gates.Kp,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity
  
  # Test Km gate
  gg_dag = gates.Km * dag(prime(gates.Km,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity
  
  # Test RX gate
  θ = π * rand()
  gate = RX(θ)
  gg_dag = gate * dag(prime(gate,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity

  # Test RY gate
  θ = π * rand()
  gate = RY(θ)
  gg_dag = gate * dag(prime(gate,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity

  # Test RZ gate
  ϕ = 2π * rand()
  gate = RZ(ϕ)
  gg_dag = gate * dag(prime(gate,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity
  
  # Test U3 gate
  angles = rand!(qc.rng, zeros(3))
  θ = π * angles[1]
  ϕ = 2π * angles[2]
  λ = 2π * angles[3]
  gate = U3(θ,ϕ,λ)
  gg_dag = gate * dag(prime(gate,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity
end

@testset "Two-qubit gates" begin
  # Test Swap gate
  sw_dag = dag(gates.Swap)
  sw_dag = setprime(sw_dag,plev=2,1)
  sw_dag = prime(sw_dag,plev=0,2)
  gg_dag = gates.Swap * sw_dag
  identity = ITensor(reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2)),inds(gg_dag))
  @test gg_dag ≈ identity

  ## Test cX gate
  #cx = cX([1,2])
  #cx_dag = dag(cx)
  #cx_dag = setprime(cx_dag,plev=2,1)
  #cx_dag = prime(cx_dag,plev=0,2)
  #gg_dag = cx * cx_dag
  #identity = ITensor(reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2)),inds(gg_dag))
  #@test gg_dag ≈ identity
  #
  ## Test reverse cX gate
  #cx = cX([2,1])
  #cx_dag = dag(cx)
  #cx_dag = setprime(cx_dag,plev=2,1)
  #cx_dag = prime(cx_dag,plev=0,2)
  #gg_dag = cx * cx_dag
  #identity = ITensor(reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2)),inds(gg_dag))
  #@test gg_dag ≈ identity
  #
  #cy = cY([1,2])
  #cy_dag = dag(cy)
  #cy_dag = setprime(cy_dag,plev=2,1)
  #cy_dag = prime(cy_dag,plev=0,2)
  #gg_dag = cy * cy_dag
  #identity = ITensor(reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2)),inds(gg_dag))
  #@test gg_dag ≈ identity
  #
  #cy = cY([2,1])
  #cy_dag = dag(cy)
  #cy_dag = setprime(cy_dag,plev=2,1)
  #cy_dag = prime(cy_dag,plev=0,2)
  #gg_dag = cy * cy_dag
  #identity = ITensor(reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2)),inds(gg_dag))
  #@test gg_dag ≈ identity
 
  # Test cZ gate
  #cz = cZ([1,2])
  #cz_dag = dag(cz)
  #cz_dag = setprime(cz_dag,plev=2,1)
  #cz_dag = prime(cz_dag,plev=0,2)
  #gg_dag = cz * cz_dag
  #identity = ITensor(reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2)),inds(gg_dag))
  #@test gg_dag ≈ identity
  #
  #cz = cZ([2,1])
  #cz_dag = dag(cz)
  #cz_dag = setprime(cz_dag,plev=2,1)
  #cz_dag = prime(cz_dag,plev=0,2)
  #gg_dag = cz * cz_dag
  #identity = ITensor(reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2)),inds(gg_dag))
  #@test gg_dag ≈ identity
end

