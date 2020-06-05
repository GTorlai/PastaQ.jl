using PastaQ
using ITensors
using HDF5
using JLD
using Test
using LinearAlgebra

function runcircuitFULL(N::Int,tensors::Array)
  """ Assumes NN gates, and for 2q gates-> [q,q+1] """
  ngates = length(tensors)
  id_mat = [1. 0.;0. 1.]
  U = 1.0
  for j in 1:N
    U = kron(U,id_mat)
  end
  for tensor in tensors
    #loop_size = N+1-length(inds(tensor))÷2
    u = 1.0
    # 1q gate
    if (length(inds(tensor)) == 2)
      site = getsitenumber(firstind(tensor,"Site"))
      for j in 1:N
        if (j == site)
          u = kron(u,fullmatrix(tensor))
        else
          u = kron(u,id_mat)
        end
      end
    #2q gate
    else
      site = getsitenumber(inds(tensor,plev=1)[1])
      for j in 1:N-1
        if (j == site)
          u = kron(u,fullmatrix(tensor))
        else
          u = kron(u,id_mat)
        end
      end
    end
    U = u * U
  end
  psi = U[:,1]
  return psi
end

@testset "qubits initialization" begin
  N = 1
  psi = qubits(N)
  @test length(psi) == 1
  @test length(inds(psi[1],"Link")) == 0
  @test fullvector(psi) ≈ [1, 0]
  N = 5
  psi = qubits(N)
  @test length(psi) == 5
  psi_vec = fullvector(psi)
  exact_vec = zeros(1<<N)
  exact_vec[1] = 1.0
  @test psi_vec ≈ exact_vec
end

@testset "runcircuit: hadamardlayer N=10" begin
  N = 10 
  gates = []
  hadamardlayer!(gates,N)
  @test length(gates) == N
  psi = qubits(N)
  tensors = compilecircuit(psi,gates)
  @test length(tensors) == N
  runcircuit!(psi,tensors)
  exact_psi = runcircuitFULL(N,tensors)
  @test exact_psi ≈ fullvector(psi)
end

@testset "runcircuit: rand1Qrotationlayer N=10" begin
  N = 10 
  gates = []
  rand1Qrotationlayer!(gates,N)
  @test length(gates) == N
  psi = qubits(N)
  tensors = compilecircuit(psi,gates)
  @test length(tensors) == N
  runcircuit!(psi,tensors)
  exact_psi = runcircuitFULL(N,tensors)
  @test exact_psi ≈ fullvector(psi)
end

@testset "runcircuit: Cx layer N=10" begin
  N = 10 
  gates = []
  Cxlayer!(gates,N,sequence = "odd") 
  @test length(gates) == N÷2
  psi = qubits(N)
  tensors = compilecircuit(psi,gates)
  @test length(tensors) == N÷2
  runcircuit!(psi,tensors)
  exact_psi = runcircuitFULL(N,tensors)
  @test exact_psi ≈ fullvector(psi)
  
  gates = []
  Cxlayer!(gates,N,sequence = "even") 
  @test length(gates) == N÷2-1
  psi = qubits(N)
  tensors = compilecircuit(psi,gates)
  @test length(tensors) == N÷2-1
  runcircuit!(psi,tensors)
  exact_psi = runcircuitFULL(N,tensors)
  @test exact_psi ≈ fullvector(psi)
end

@testset "runcircuit: random quantum circuit" begin
  N = 10
  depth = 8
  gates = randomquantumcircuit(N,depth)
  ngates = N*depth + depth÷2 * (N-1)
  @test length(gates) == ngates
  psi = qubits(N)
  tensors = compilecircuit(psi,gates)
  @test length(tensors) == ngates
  runcircuit!(psi,tensors)
  exact_psi = runcircuitFULL(N,tensors)
  @test exact_psi ≈ fullvector(psi)
end

#@testset "runcircuit: randomRnCx N=10" begin
#  data = load("testdata/quantumcircuit_testunitary_randomRnCx.jld")
#  N = data["N"]
#  gate_list = data["gates"]
#  exact_unitary = data["full_unitary"]
#  exact_psi     = data["full_psi"]  
#  psi = qubits(N)
#  gates = compilecircuit(psi,gate_list)
#  runcircuit!(psi,gates)
#  @test exact_psi ≈ fullvector(psi,order="native")
#end
#
#@testset "runcircuit: hadamardlayer N=10" begin
#  data = load("testdata/quantumcircuit_testunitary_hadamardlayer.jld")
#  N = data["N"]
#  gate_list = data["gates"]
#  exact_unitary = data["full_unitary"]
#  exact_psi     = data["full_psi"]  
#  psi = qubits(N)
#  gates = compilecircuit(psi,gate_list)
#  runcircuit!(psi,gates)
#  @test exact_psi ≈ fullvector(psi)
#end

#@testset "runcircuit: rand1Qrotationlayer N=10" begin
#  data = load("testdata/quantumcircuit_testunitary_rand1Qrotationlayer.jld")
#  N = data["N"]
#  gate_list = data["gates"]
#  exact_unitary = data["full_unitary"]
#  exact_psi     = data["full_psi"]  
#  psi = qubits(N)
#  gates = compilecircuit(psi,gate_list)
#  runcircuit!(psi,gates)
#  @test exact_psi ≈ fullvector(psi,order="native")
#end
##
##@testset "circuit initialization" begin
##  N=5
##  U = initializecircuit(N)
##  @test length(U) == 5
##  identity = itensor(reshape([1 0;0 1],(1,2,2)),inds(U[1]))
##  @test U[1] ≈ identity
##  for s in 2:N-1
##    identity = itensor(reshape([1 0;0 1],(1,1,2,2)),inds(U[s]))
##    @test U[s] ≈ identity
##  end
##  identity = itensor(reshape([1 0;0 1],(1,2,2)),inds(U[N]))
##  @test U[N] ≈ identity
##end
#
