using PastaQ
using ITensors
using Test
using Random

@testset "nqubits/nlayers/ngates" begin
  @test nqubits(("H", 2)) == 2
  @test nqubits(("CX", (2, 5))) == 5

  for i in 1:1000
    depth = 4
    N = rand(3:50)
    gates = randomcircuit(N, depth; twoqubitgates="CX", onequbitgates="Rn")
    n = nqubits(gates)
    @test N == n
    @test PastaQ.nlayers(gates) == depth
    @test PastaQ.ngates(gates) == depth ÷ 2 * (2 * N ÷ 2 - 1 + 2 * N)
    gates = randomcircuit(N, depth; twoqubitgates="CX", onequbitgates="Rn", layered=false)
    n = nqubits(gates)
    @test N == n
    @test PastaQ.nlayers(gates) == 1
    @test PastaQ.ngates(gates) == depth ÷ 2 * (2 * N ÷ 2 - 1 + 2 * N)
  end

  N = 3
  
  # MPS
  circuit = randomcircuit(N,4)
  ψ = runcircuit(circuit; full_representation = true) 
  @test nqubits(ψ) == N
  # MPO
  U = runcircuit(circuit; full_representation = true, process = true) 
  @test nqubits(U) == N
  # LPDO DM
  ρ = prod(randomstate(N; ξ = 2)) 
  @test nqubits(ρ) == N
  
  # MPO Choi
  Λ = runcircuit(circuit; noise = ("DEP",(p=0.01,)), full_representation = true, process = true) 
  @test nqubits(Λ) == N
  Λ = prod(randomprocess(N; ξ = 2)) 
  @test nqubits(Λ) == N
end
@testset "pre-defined ciruits" begin
  N = 10
  @test length(qft(N)) == sum(1:(N - 1)) + N
  @test length(qft(N; inverse=true)) == sum(1:(N - 1)) + N
  @test length(ghz(N)) == N
end

@testset "gatelayer" begin
  N = 10

  layer = gatelayer("X", N)
  @test length(layer) == N
  @test all(x -> x == "X", first.(layer))

  layer = gatelayer("X", 1:2:N)
  @test length(layer) == N ÷ 2

  layer = gatelayer("X", [1, 3, 5, 7])
  @test length(layer) == 4

  layer = gatelayer("Rx", N; θ=0.1)
  @test haskey.(last.(layer), :θ) == ones(N)
  @test values.(last.(layer)) == [(0.1,) for _ in 1:N]

  qarray = PastaQ.lineararray(N)
  layer = gatelayer("CX", qarray[1])
  @test length(layer) == length(qarray[1])
  @test all(x -> x == "CX", first.(layer))
  layer = gatelayer("CX", qarray[2])
  @test length(layer) == length(qarray[2])
  @test all(x -> x == "CX", first.(layer))
end

@testset "randommparams" begin
  N = 10
  pars = PastaQ.randomparams("Rx")
  @test haskey(pars, :θ)
  pars = PastaQ.randomparams("Ry")
  @test haskey(pars, :θ)
  pars = PastaQ.randomparams("Rz")
  @test haskey(pars, :ϕ)
  pars = PastaQ.randomparams("Rn")
  @test haskey(pars,:θ)
  @test haskey(pars,:ϕ)
  @test haskey(pars,:λ)
  
  pars = PastaQ.randomparams("RandomUnitary",2)
  @test haskey(pars,:random_matrix)
  @test size(pars[:random_matrix]) == (4,4)
  @test pars[:random_matrix] isa Matrix{ComplexF64}
end

@testset "randomlayer" begin
  N = 10
  layer = randomlayer("Rn", N)
  @test length(layer) == N
  @test all(x -> x == "Rn", first.(layer))

  qarray = PastaQ.lineararray(N)
  layer = randomlayer("RandomUnitary", qarray[1])
  @test length(layer) == length(qarray[1])
  @test all(x -> x == "RandomUnitary", first.(layer))
  layer = randomlayer("RandomUnitary", qarray[2])
  @test length(layer) == length(qarray[2])
  @test all(x -> x == "RandomUnitary", first.(layer))
end

@testset "random circuits" begin
  N = 30
  depth = 10
  circuit = randomcircuit(N, depth; twoqubitgates="RandomUnitary")
  @test length(circuit) == depth
  for d in 1:depth
    @test all(x -> x == "RandomUnitary", first.(circuit[depth]))
  end

  circuit = randomcircuit(N, depth; twoqubitgates="CX")
  @test PastaQ.nlayers(circuit) == depth
  for d in 1:depth
    @test all(x -> x == "CX", first.(circuit[depth]))
  end

  circuit = randomcircuit(N, depth; twoqubitgates="CX", onequbitgates="Rn")
  @test size(circuit, 1) == depth

  circuit = randomcircuit(
    N, depth; twoqubitgates="RandomUnitary", onequbitgates=["Rn", "X"]
  )
  @test size(circuit, 1) == depth

  Lx = 5
  Ly = 5
  circuit = randomcircuit(Lx,Ly, depth; twoqubitgates="RandomUnitary", onequbitgates=["Rn", "X"])
  @test nqubits(circuit) == Lx*Ly
  
  Lx = 5
  Ly = 5
  circuit = randomcircuit(Lx,Ly, depth; twoqubitgates="RandomUnitary", onequbitgates=["Rn", "X"], rotated = true)
  @test nqubits(circuit) == Lx*Ly
end

@testset "dag circuit" begin

  N = 2
  depth = 4

  circuit = randomcircuit(N, depth; twoqubitgates = "CX", onequbitgates = "Rn")
  U = PastaQ.array(runcircuit(circuit; process = true))
  dagcircuit = dag(circuit)
  V = PastaQ.array(runcircuit(dagcircuit; process = true))
  @test U ≈ V'
  
  circuit = randomcircuit(N, depth; twoqubitgates = "RandomUnitary", onequbitgates = "Rn", layered = false)
  U = PastaQ.array(runcircuit(circuit; process = true))
  dagcircuit = dag(circuit)
  V = PastaQ.array(runcircuit(dagcircuit; process = true))
  @test U ≈ V'
  
  dagdagcircuit = dag(dagcircuit)
  Up = PastaQ.array(runcircuit(dagdagcircuit; process = true))
  @test U ≈ Up

end

