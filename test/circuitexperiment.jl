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

@testset "Quantum state preparation" begin
  N = 5
  testdata = load("testdata/circuitexp_prepPauli6.jld")
  gates = QuantumGates()
  qc = QuantumCircuit(N=N)
  experiment = CircuitExperiment(N=N)
  BuildPreparationBases!(experiment,gates,prep_id="pauli6")
  for s in 1:size(testdata["input_states"])[1]
    state = testdata["input_states"][s,:]
    psi0 = PrepareState(qc,experiment,state)
    psi_vec = FullVector(psi0)
    exact_psi = ITensor(testdata["psi"][s,:],inds(psi_vec))
    @test psi_vec ≈ exact_psi
  end
end


@testset "Run circuit with input=Pauli6 and output=COMP" begin
  N = 5
  testdata = load("testdata/circuitexp_prepPauli6_measCOMP.jld")
  gates = QuantumGates()
  qc = QuantumCircuit(N=N)
  experiment = CircuitExperiment(N=N)
  LoadQuantumCircuit(qc,gates,testdata["gate_list"],cutoff=1e-10)
  BuildPreparationBases!(experiment,gates,prep_id="pauli6")
  full_unitary  = FullMatrix(qc.U)
  exact_unitary = ITensor(testdata["full_unitary"],inds(full_unitary))
  @test full_unitary ≈ exact_unitary
  for s in 1:size(testdata["input_states"])[1]
    state = testdata["input_states"][s,:]
    psi0 = PrepareState(qc,experiment,state)
    psi_out = ApplyCircuit(qc,psi0)
    psi_vec = FullVector(psi_out)
    exact_psi = ITensor(testdata["psi"][s,:],inds(psi_vec))
    @test psi_vec ≈ exact_psi
  end
end

@testset "Run circuit with input=COMP and output=Pauli6" begin
  N = 5
  testdata = load("testdata/circuitexp_prepCOMP_measPauli6.jld")
  gates = QuantumGates()
  qc = QuantumCircuit(N=N)
  experiment = CircuitExperiment(N=N)
  LoadQuantumCircuit(qc,gates,testdata["gate_list"],cutoff=1e-10)
  full_unitary  = FullMatrix(qc.U)
  exact_unitary = ITensor(testdata["full_unitary"],inds(full_unitary))
  @test full_unitary ≈ exact_unitary
  BuildMeasurementBases!(experiment,gates,prep_id="pauli6")
  for s in 1:size(testdata["meas_bases"])[1]
    basis = testdata["meas_bases"][s,:]
    psi0 = InitializeQubits(qc) 
    psi = ApplyCircuit(qc,psi0)
    psi_out = RotateMeasurementBasis!(psi,qc,experiment,basis)
    psi_vec = FullVector(psi_out)
    exact_psi = ITensor(testdata["psi"][s,:],inds(psi_vec))
    @test psi_vec ≈ exact_psi
  end
end

@testset "Run circuit with input=COMP and output=Pauli6" begin
  N = 5
  testdata = load("testdata/circuitexp_prepPauli6_measPauli6.jld")
  gates = QuantumGates()
  qc = QuantumCircuit(N=N)
  experiment = CircuitExperiment(N=N)
  LoadQuantumCircuit(qc,gates,testdata["gate_list"],cutoff=1e-10)
  full_unitary  = FullMatrix(qc.U)
  exact_unitary = ITensor(testdata["full_unitary"],inds(full_unitary))
  @test full_unitary ≈ exact_unitary
  BuildPreparationBases!(experiment,gates,prep_id="pauli6")
  BuildMeasurementBases!(experiment,gates,prep_id="pauli6")
  for s in 1:size(testdata["input_states"])[1]
    state = testdata["input_states"][s,:]
    basis = testdata["meas_bases"][s,:]
    psi0 = PrepareState(qc,experiment,state)
    psi = ApplyCircuit(qc,psi0)
    psi_out = RotateMeasurementBasis!(psi,qc,experiment,basis)
    psi_vec = FullVector(psi_out)
    exact_psi = ITensor(testdata["psi"][s,:],inds(psi_vec))
    @test psi_vec ≈ exact_psi
  end
end

