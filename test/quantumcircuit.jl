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

@testset "Circuit MPO" begin
  N=5
  testdata = load(string("test_data_N",N,".jld"))
  qgates = QuantumGates()
  qc = QuantumCircuit(N=N)
  LoadQuantumCircuit(qc,qgates,testdata["gate_list"])
  full_unitary  = FullMatrix(qc.U)
  exact_unitary = ITensor(testdata["full_unitary"],inds(full_unitary))
  @show norm(full_unitary-exact_unitary)
  #@test full_unitary ≈ exact_unitary atol=1e-6
end



#state = InitializeQubits(qc)
#vector = FullVector(state)
#state_id = [6,1,1]
#mps_state = StatePreparation(qc,qgates,state_id)
#vector = FullVector(mps_state)
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
