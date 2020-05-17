include("../src/tnqpt.jl")
using Main.TNQPT
using HDF5, JLD
using ITensors
using Test
using LinearAlgebra

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
  @test gg_dag ≈ identity atol=1e-8
  
  gg_dag = qgates.Y * dag(prime(qgates.Y,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity atol=1e-8
  
  gg_dag = qgates.Z * dag(prime(qgates.Z,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity atol=1e-8
  
  gg_dag = qgates.H * dag(prime(qgates.H,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity atol=1e-8
  
  gg_dag = qgates.K * dag(prime(qgates.K,plev=0,2))
  identity = ITensor(Matrix{ComplexF64}(I, 2, 2),inds(gg_dag))
  @test gg_dag ≈ identity atol=1e-8
  
  gg_dag = qgates.cX * dag(prime(qgates.cX,plev=0,2))
  identity = ITensor(reshape(Matrix{ComplexF64}(I, 4, 4),(2,2,2,2)),inds(gg_dag))
  @test gg_dag ≈ identity atol=1e-8
  
end

@testset "Single-qubit circuit MPO" begin
  N=5
  testdata = load(string("test_data_N",N,"_singlequbit.jld"))
  qgates = QuantumGates()
  qc = QuantumCircuit(N=N)
  LoadQuantumCircuit(qc,qgates,testdata["gate_list"])
  full_unitary  = FullMatrix(qc.U)
  exact_unitary = ITensor(testdata["full_unitary"],inds(full_unitary))
  #@show norm(full_unitary-exact_unitary)
  #" The norm checks out (with my TN module), but comparison fails"
  @test full_unitary ≈ exact_unitary #atol=1e-10
end

@testset "Two-qubit circuit MPO (no truncation)" begin
  N=5
  testdata = load(string("test_data_N",N,"_twoqubit.jld"))
  qgates = QuantumGates()
  qc = QuantumCircuit(N=N)
  LoadQuantumCircuit(qc,qgates,testdata["gate_list"])
  full_unitary  = FullMatrix(qc.U)
  exact_unitary = ITensor(testdata["full_unitary"],inds(full_unitary))
  @test full_unitary ≈ exact_unitary #atol=1e-10
  @show maxlinkdim(qc.U)
end


@testset "Two-qubit circuit MPO (truncation)" begin
  N=10
  testdata = load(string("test_data_N",N,"_twoqubit.jld"))
  qgates = QuantumGates()
  #cutoff_list = [1e-14,1e-12,1e-10,1e-8,1e-6,1e-4]
  #for c in 1:length(cutoff_list)
  #cutoff = cutoff_list[c]
  cutoff=1e-10
  qc = QuantumCircuit(N=N)
  LoadQuantumCircuit(qc,qgates,testdata["gate_list"],cutoff=cutoff)
  full_unitary  = FullMatrix(qc.U)
  exact_unitary = ITensor(testdata["full_unitary"],inds(full_unitary))
  @test full_unitary ≈ exact_unitary #atol=1e-10
  #println("cutoff = ",cutoff," chi = ",maxlinkdim(qc.U))
  #end
end










#N=4
#qgates = QuantumGates()
#qc = QuantumCircuit(N=N)
#gate = cX([1,2])
#ApplyTwoQubitGate!(qc.U,gate,[2 3])
#full_unitary  = FullMatrix(qc.U)
#@show full_unitary




##testdata = load(string("test_data_N",N,"_singlequbit.jld"))
##LoadQuantumCircuit(qc,qgates,testdata["gate_list"])
##@show qc.U
#ApplyTwoQubitGate!(qc.U,qgates.cX,[1 2])
#ApplyTwoQubitGate!(qc.U,qgates.cX,[2 3])
#ApplyTwoQubitGate!(qc.U,qgates.cX,[3 4])
#ApplyTwoQubitGate!(qc.U,qgates.cX,[4 5])
##@show qc.U
##state = InitializeQubits(qc)
##vector = FullVector(state)
##state_id = [6,1,1]
##mps_state = StatePreparation(qc,qgates,state_id)
##vector = FullVector(mps_state)
#@show vector

#@show isempty(qc.gate_list)
#RandomSingleQubitLayer!()
#RunQuantumCircuit(qc,qgates)


#N=5
#testdata = load(string("test_data_N",N,".jld"))
#@show keys(testdata)
#N=3
#qgates = QuantumGates()
#qc = QuantumCircuit(N=N)
