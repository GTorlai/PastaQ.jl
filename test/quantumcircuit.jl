include("../src/PastaQ.jl")
using Main.PastaQ
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

@testset "Single-qubit circuit MPO" begin
  N=10
  testdata = load("testdata/quantumcircuit_unitary_singlequbit.jld")
  gates = QuantumGates()
  qc = QuantumCircuit(N=N)
  LoadQuantumCircuit(qc,gates,testdata["gate_list"])
  full_unitary  = FullMatrix(qc.U)
  exact_unitary = ITensor(testdata["full_unitary"],inds(full_unitary))
  @test full_unitary ≈ exact_unitary
end

@testset "Two-qubit circuit MPO" begin
  N=10
  testdata = load("testdata/quantumcircuit_unitary_twoqubit.jld")
  gates = QuantumGates()
  cutoff=1e-10
  qc = QuantumCircuit(N=N)
  LoadQuantumCircuit(qc,gates,testdata["gate_list"],cutoff=cutoff)
  full_unitary  = FullMatrix(qc.U)
  exact_unitary = ITensor(testdata["full_unitary"],inds(full_unitary))
  @test full_unitary ≈ exact_unitary
end

@testset "Run circuit with in the computational basis" begin
  N=5
  testdata = load("testdata/quantumcircuit_run_computational.jld")
  gates = QuantumGates()
  qc = QuantumCircuit(N=N)
  cutoff=1e-10
  LoadQuantumCircuit(qc,gates,testdata["gate_list"],cutoff=cutoff)
  full_unitary  = FullMatrix(qc.U)
  exact_unitary = ITensor(testdata["full_unitary"],inds(full_unitary))
  @test full_unitary ≈ exact_unitary
  psi = InitializeQubits(qc)
  psi_out = ApplyCircuit(qc,psi)
  psi_vec = FullVector(psi_out)
  exact_psi = ITensor(testdata["psi"],inds(psi_vec))
  @test psi_vec ≈ exact_psi
end

