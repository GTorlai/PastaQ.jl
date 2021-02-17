using PastaQ
using PastaQ.ITensors
using Test
using LinearAlgebra
using Random

@testset "surface code contructor" begin
  d = 15
  code = SurfaceCode(d)
  @test d ≈ code.d
  @test d ≈ distance(code)
end

@testset "qubit at coordinate" begin
  d = 15
  code = SurfaceCode(d)
  r = PastaQ.coordinates_ofQubits(code)
  # horizontal qubits
  for y in 1:2:2*d-1
    for x in 1:2:2*d-1
      q = PastaQ.qubit_at(code,x,y)
      @test r[q] == (x,y)
    end
  end
  # horizontal qubits
  for y in 2:2:2*d-1
    for x in 2:2:2*d-1
      q = PastaQ.qubit_at(code,x,y)
      @test r[q] == (x,y)
    end
  end
end

@testset "stabilizers at coordinate" begin
  d = 15
  code = SurfaceCode(d)
  # X stabilizers 
  c = 1
  for y in 1:2:2*d-1
    for x in 2:2:2*d-2
      X = PastaQ.stabX_at(code,x,y)
      @test X == c
      r = code.Xcoord[X]
      @test r[1] == x
      @test r[2] == y
      c+= 1
    end
  end
  # X stabilizers 
  c = 1
  for y in 2:2:2*d-1
    for x in 1:2:2*d-1
      Z = PastaQ.stabZ_at(code,x,y)
      @test Z == c
      r = code.Zcoord[Z]
      @test r[1] == x
      @test r[2] == y
      c+= 1
    end
  end
end

@testset "logical operator" begin
  d = 5
  code = SurfaceCode(d)

  # check that logical operators do not trigger a syndrome
  logicalI = PastaQ.logicaloperator("I",code)
  logicalZ = PastaQ.logicaloperator("Z",code)
  logicalX = PastaQ.logicaloperator("X",code)
  logicalY = PastaQ.logicaloperator("Y",code)
  
  @test logicalI == PastaQ.logicaloperator((0,0),code)  
  @test logicalZ == PastaQ.logicaloperator((0,1),code)
  @test logicalX == PastaQ.logicaloperator((1,0),code)
  @test logicalY == PastaQ.logicaloperator((1,1),code)
                                                     
  @test logicalI == PastaQ.logicaloperator([0,0],code)
  @test logicalZ == PastaQ.logicaloperator([0,1],code)
  @test logicalX == PastaQ.logicaloperator([1,0],code)
  @test logicalY == PastaQ.logicaloperator([1,1],code)
  s = syndrome(logicalI, code)
  @test all(s[:Z] .== 0)
  @test all(s[:X] .== 0)
  s = syndrome(logicalX, code)
  @test all(s[:Z] .== 0)
  @test all(s[:X] .== 0)
  s = syndrome(logicalZ, code)
  @test all(s[:Z] .== 0)
  @test all(s[:X] .== 0)
  s = syndrome(logicalY, code)
  @test all(s[:Z] .== 0)
  @test all(s[:X] .== 0)
end

@testset "syndromes" begin
  d = 15
  nsamples = 100
  code = SurfaceCode(d)
  N = nqubits(code)
  E = depolarizingerror(code, nsamples; p = 0.3) 
  S = syndrome(E, code)
  
  for i in 1:nsamples
    e = E[i]; s = S[i]
    @test length(e) == N 
    @test length(s[:X]) == d*(d-1)
    # apply random stabilizers
    for j in 1:10
      random_stabXsupport = PastaQ.qubits_atStabX(code,rand(1:d*(d-1))) 
      random_stabZsupport = PastaQ.qubits_atStabZ(code,rand(1:d*(d-1)))
      random_stabX = PastaQ.support_to_pauli(code,random_stabXsupport)
      random_stabZ = PastaQ.support_to_pauli(code,random_stabZsupport)
      random_stab   = [(random_stabX[j],random_stabZ[j]) for j in 1:nqubits(code)] 
                                    
      ẽ = e ⊙ random_stab
      s̃ = syndrome(ẽ, code)
      @test s == s̃
    end 
  end
end

@testset "reference pauli" begin
  d = 15
  nsamples = 100
  code = SurfaceCode(d)
  E = depolarizingerror(code, nsamples; p = 0.3) 
  S = syndrome(E, code)
  
  #E, S = getsamples(code, nsamples; error_probability = probs)  

  for n in 1:nsamples
    f = PastaQ.purepaulierror(S[n], code)
    s̃ = syndrome(f, code)
    @test S[n] == s̃
    effective_pauli = E[n] ⊙ f
    s_trivial = syndrome(effective_pauli, code)
    @test all(s_trivial[:Z] .== 0)
    @test all(s_trivial[:X] .== 0)
  end
end
@testset "Wilson loops" begin
  d = 5
  code = SurfaceCode(d)
  nsamples = 100
  E = depolarizingerror(code, nsamples; p = 0.3) 
  S = syndrome(E, code)

  for n in 1:nsamples
    e = E[n]
    s = S[n]
    f = PastaQ.purepaulierror(s, code)
    r = e ⊙ f
    s̃ = syndrome(r, code)
    @test all(s̃[:Z] .== 0)
    @test all(s̃[:X] .== 0)
    wX,wZ = PastaQ.Wilsonloops(r, code)

    # apply logical X
    logicalX = PastaQ.logicaloperator("X", code)
    L = r ⊙ logicalX
    s̃ = syndrome(L, code)
    @test all(s̃[:Z] .== 0)
    @test all(s̃[:X] .== 0)
    w̃X, w̃Z = PastaQ.Wilsonloops(L, code)
    @test wX == w̃X ⊻ 1
    @test wZ == w̃Z
    
    # apply logical X
    logicalZ = PastaQ.logicaloperator("Z", code)
    L = r ⊙ logicalZ
    s̃ = syndrome(L, code)
    @test all(s̃[:Z] .== 0)
    @test all(s̃[:X] .== 0)
    w̃X, w̃Z = PastaQ.Wilsonloops(L, code)
    @test wX == w̃X
    @test wZ == w̃Z ⊻ 1
  
    # apply logical X
    logicalY = PastaQ.logicaloperator("Y", code)
    L = r ⊙ logicalY
    s̃ = syndrome(L, code)
    @test all(s̃[:Z] .== 0)
    @test all(s̃[:X] .== 0)
    w̃X, w̃Z = PastaQ.Wilsonloops(L, code)
    @test wX == w̃X ⊻ 1
    @test wZ == w̃Z ⊻ 1
  end
end

