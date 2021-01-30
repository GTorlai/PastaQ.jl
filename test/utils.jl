using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random

@testset "array" begin
  N = 5
  ψ = qubits(N)
  ψvec = array(ψ)
  @test size(ψvec) == (1<<N,)
  
  ρ = qubits(N; mixed = true)
  ρmat = array(ρ)
  @test size(ρmat) == (1<<N,1<<N)
  
  ρ = randomstate(N; mixed = true)
  ρmat = array(ρ)
  @test size(ρmat) == (1<<N,1<<N)

  U = randomprocess(N)
  Umat = array(U)
  @test size(Umat) == (1<<N,1<<N)
  
  N = 3
  Λ = randomprocess(N; mixed = true)
  #Λmat = array(Λ)
end

@testset "hilbertspace" begin
  N = 5
  ψ = qubits(N)
  ρ = qubits(ψ; mixed = true)
  Λ = randomstate(ψ; mixed = true)

  @test PastaQ.hilbertspace(ψ) == siteinds(ψ)
  @test PastaQ.hilbertspace(ψ) == PastaQ.hilbertspace(ρ)
  @test PastaQ.hilbertspace(ψ) == PastaQ.hilbertspace(Λ)

end


#@testset "replace hilbert space" begin
#
#  N = 5
#  # REF is MPS
#  REF = qubits(N)
#  ψ = qubits(N)
#  ρ = qubits(N; mixed = true)
#  ϱ = randomstate(N; mixed = true)
#  Λ = randomprocess(N; mixed = true)
#
#  PastaQ.replace_hilbertspace!(ψ,REF) 
#  @test siteinds(ψ) == siteinds(REF)
#
#  PastaQ.replace_hilbertspace!(ρ,REF) 
#  @test firstsiteinds(ρ) == siteinds(REF)
#
#
#
#
#end



@testset "numberofqubits" begin
  
  @test numberofqubits(("H",2)) == 2
  @test numberofqubits(("CX",(2,5))) == 5

  for i in 1:10
    depth = 4
    N = rand(2:50)
    gates = randomcircuit(N, depth)
    n = numberofqubits(gates)
    @test N == n
  end
end


