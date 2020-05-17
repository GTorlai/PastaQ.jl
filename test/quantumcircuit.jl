include("../src/tnqpt.jl")
using Main.TNQPT
using HDF5, JLD
using ITensors
using Test
using LinearAlgebra

function FullVector(mps::MPS)
  vector = mps[1] * mps[2]
  C = combiner(inds(vector,tags="s=1")[1],inds(vector,tags="s=2")[1],tags="comb")
  vector = vector * C
  for j in 3:length(mps)
    vector = vector * mps[j]
    C = combiner(inds(vector,tags="comb")[1],inds(vector,tags="s=$j")[1],tags="comb")
    vector = vector * C
  end
  return vector
end

function FullMatrix(mpo::MPO)
  matrix = mpo[1] * mpo[2]
  Cb = combiner(inds(matrix,tags="s=1",plev=0)[1],inds(matrix,tags="s=2",plev=0)[1],tags="bra")
  Ck = combiner(inds(matrix,tags="s=1",plev=1)[1],inds(matrix,tags="s=2",plev=1)[1],tags="ket")
  matrix = matrix * Cb * Ck
  for j in 3:length(mpo)
    matrix = matrix * mpo[j]
    Cb = combiner(inds(matrix,tags="bra")[1],inds(matrix,tags="s=$j",plev=0)[1],tags="bra")
    Ck = combiner(inds(matrix,tags="ket")[1],inds(matrix,tags="s=$j",plev=1)[1],tags="ket")
    matrix = matrix * Cb * Ck
  end
  return matrix
end

@testset "Quantum gates" begin
  qgates = QuantumGates()
  
  gg_dag = qgates.X * dag(prime(qgates.X,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity
  
  gg_dag = qgates.Y * dag(prime(qgates.Y,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity 
  
  gg_dag = qgates.Z * dag(prime(qgates.Z,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity
  
  gg_dag = qgates.H * dag(prime(qgates.H,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity
  
  gg_dag = qgates.Kp * dag(prime(qgates.Kp,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity
  
  gg_dag = qgates.Km * dag(prime(qgates.Km,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity
  
  cx = cX([1,2])
  cx_dag = dag(cx)
  cx_dag = setprime(cx_dag,plev=2,1)
  cx_dag = prime(cx_dag,plev=0,2)
  gg_dag = cx * cx_dag
  identity = ITensor(reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2)),inds(gg_dag))
  @test gg_dag ≈ identity
  
end

@testset "Single-qubit circuit MPO" begin
  N=10
  testdata = load("testdata/quantumcircuit_unitary_singlequbit.jld")
  qgates = QuantumGates()
  qc = QuantumCircuit(N=N)
  LoadQuantumCircuit(qc,qgates,testdata["gate_list"])
  full_unitary  = FullMatrix(qc.U)
  exact_unitary = ITensor(testdata["full_unitary"],inds(full_unitary))
  #@show norm(full_unitary-exact_unitary)
  #" The norm checks out (with my TN module), but comparison fails"
  @test full_unitary ≈ exact_unitary #atol=1e-10
end

@testset "Two-qubit circuit MPO" begin
  N=10
  testdata = load("testdata/quantumcircuit_unitary_twoqubit.jld")
  qgates = QuantumGates()
  cutoff=1e-10
  qc = QuantumCircuit(N=N)
  LoadQuantumCircuit(qc,qgates,testdata["gate_list"],cutoff=cutoff)
  full_unitary  = FullMatrix(qc.U)
  exact_unitary = ITensor(testdata["full_unitary"],inds(full_unitary))
  @test full_unitary ≈ exact_unitary #atol=1e-10
  #println("cutoff = ",cutoff," chi = ",maxlinkdim(qc.U))
end

@testset "Run circuit with in the computational basis" begin
  N=5
  testdata = load("testdata/quantumcircuit_run_computational.jld")
  qgates = QuantumGates()
  qc = QuantumCircuit(N=N)
  cutoff=1e-10
  LoadQuantumCircuit(qc,qgates,testdata["gate_list"],cutoff=cutoff)
  full_unitary  = FullMatrix(qc.U)
  exact_unitary = ITensor(testdata["full_unitary"],inds(full_unitary))
  @test full_unitary ≈ exact_unitary
  psi = InitializeQubits(qc)
  psi_out = ApplyCircuit(qc,psi)
  #@show psi_out
  psi_vec = FullVector(psi_out)
  exact_psi = ITensor(testdata["psi"],inds(psi_vec))
  #@show psi_vec
  #@show exact_psi
  @test psi_vec ≈ exact_psi
end

