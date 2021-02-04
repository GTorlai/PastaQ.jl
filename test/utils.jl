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

@testset "choi tags and MPO/MPS conversion" begin
  N = 4
  circuit = randomcircuit(4,4)

  U = runcircuit(circuit; process = true)
  ρ = PastaQ.choimatrix(PastaQ.hilbertspace(U),circuit; noise = ("DEP",(p=0.01,)))
  Λ = randomprocess(4; mixed = true)

  @test PastaQ._haschoitags(U) == false
  @test PastaQ._haschoitags(ρ) == true
  @test PastaQ._haschoitags(Λ) == true
  
  Ψ = PastaQ._choitags(U)
  @test hastags(inds(Ψ[1]),"Input") == true 
  @test hastags(inds(Ψ[1]),"Output") == true 
  @test plev(firstind(Ψ[1],tags="Input")) == 0
  @test plev(firstind(Ψ[1],tags="Output")) == 0
  
  V = PastaQ._mpotags(Ψ)
  @test hastags(inds(V[1]),"Input") == false 
  @test hastags(inds(V[1]),"Output") == false 
  @test plev(inds(V[1],tags="Qubit")[1]) == 1
  @test plev(inds(V[1],tags="Qubit")[2]) == 0

  Ψ = PastaQ._unitaryMPO_to_choiMPS(U)
  @test Ψ isa MPS 
  @test hastags(inds(Ψ[1]),"Input") == true 
  @test hastags(inds(Ψ[1]),"Output") == true 
  @test plev(firstind(Ψ[1],tags="Input")) == 0
  @test plev(firstind(Ψ[1],tags="Output")) == 0
  L = PastaQ._unitaryMPO_to_choiMPS(LPDO(U))
  @test L isa LPDO{MPS}
  @test hastags(inds(L.X[1]),"Input") == true 
  @test hastags(inds(L.X[1]),"Output") == true 
  @test plev(firstind(L.X[1],tags="Input")) == 0
  @test plev(firstind(L.X[1],tags="Output")) == 0
  
  V = PastaQ._choiMPS_to_unitaryMPO(Ψ)
  @test V isa MPO
  @test hastags(inds(V[1]),"Input") == false 
  @test hastags(inds(V[1]),"Output") == false 
  @test plev(inds(V[1],tags="Qubit")[1]) == 1
  @test plev(inds(V[1],tags="Qubit")[2]) == 0
  
  X = PastaQ._choiMPS_to_unitaryMPO(L)
  @test X isa LPDO{MPO}
  @test hastags(inds(X.X[1]),"Input") == false 
  @test hastags(inds(X.X[1]),"Output") == false 
  @test plev(inds(X.X[1],tags="Qubit")[1]) == 1
  @test plev(inds(X.X[1],tags="Qubit")[2]) == 0
  
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


