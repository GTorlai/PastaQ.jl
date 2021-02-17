using PastaQ
using PastaQ.ITensors
using Test
using LinearAlgebra
using Random

@testset "error chains" begin
  d = 15
  probs = (pX = 0.1, pY = 0.1, pZ = 0.1)
  code = SurfaceCode(d)
  
  e = PastaQ.paulierror(10; error_probability = probs)
  @test length(e) == 10
  e = PastaQ.paulierror(code; error_probability = probs)
  @test length(e) == nqubits(code)
  nchains = 50
  E = PastaQ.paulierror(code,nchains; error_probability = probs)
  @test length(E) == nchains
  
  e = PastaQ.bitfliperror(10; p = 0.1)
  @test length(e) == 10
  e = PastaQ.bitfliperror(code; p = 0.1)
  @test length(e) == nqubits(code)
  nchains = 50
  E = PastaQ.bitfliperror(code,nchains; p = 0.1)
  @test length(E) == nchains
  
  e = PastaQ.phasefliperror(10; p = 0.1)
  @test length(e) == 10
  e = PastaQ.phasefliperror(code; p = 0.1)
  @test length(e) == nqubits(code)
  nchains = 50
  E = PastaQ.phasefliperror(code,nchains; p = 0.1)
  @test length(E) == nchains
 
  e = PastaQ.depolarizingerror(10; p = 0.1)
  @test length(e) == 10
  e = PastaQ.depolarizingerror(code; p = 0.1)
  @test length(e) == nqubits(code)
  nchains = 50
  E = PastaQ.depolarizingerror(code,nchains; p = 0.1)
  @test length(E) == nchains
end

@testset "codetensors" begin
  d = 15
  code = SurfaceCode(d)
  pX = 0.05
  pY = 0.01
  pZ = 0.15
  pI = 1 - pX - pY - pZ
  probs = (pX = pX, pY = pY, pZ = pZ)  
  
  ## 2 legs (X / Z)
  H = PastaQ.codetensor(probs; Xrank = 1, Zrank = 1)
  for (x,z) in Iterators.product(0:1,0:1)
    for (X,Z) in Iterators.product(0:1,0:1)
      el = ((x⊻X == 0) && (z⊻Z == 0) ? pI :
            (x⊻X == 1) && (z⊻Z == 0) ? pX :
            (x⊻X == 0) && (z⊻Z == 1) ? pZ :
                                       pY)
      @test H[X+1,Z+1][x+1,z+1] ≈ el
    end
  end
  
  # 3 legs (X / X / Z)
  H = PastaQ.codetensor(probs; Xrank = 2, Zrank = 1)
  for (x1,x2,z) in Iterators.product(0:1,0:1,0:1)
    for (X,Z) in Iterators.product(0:1,0:1)
      el = ((x1⊻x2⊻X == 0) && (z⊻Z == 0) ? pI :
            (x1⊻x2⊻X == 1) && (z⊻Z == 0) ? pX :
            (x1⊻x2⊻X == 0) && (z⊻Z == 1) ? pZ :
                                       pY)
      @test H[X+1,Z+1][x1+1,x2+1,z+1] ≈ el
    end
  end

  # 3 legs (X / Z / Z)
  H = PastaQ.codetensor(probs; Xrank = 1, Zrank = 2)
  for (x,z1,z2) in Iterators.product(0:1,0:1,0:1)
    for (X,Z) in Iterators.product(0:1,0:1)
      el = ((x⊻X == 0) && (z1⊻z2⊻Z == 0) ? pI :
            (x⊻X == 1) && (z1⊻z2⊻Z == 0) ? pX :
            (x⊻X == 0) && (z1⊻z2⊻Z == 1) ? pZ :
                                       pY)
      @test H[X+1,Z+1][x+1,z1+1,z2+1] ≈ el
    end
  end
  # 4 legs (X / X / Z / Z)
  H = PastaQ.codetensor(probs; Xrank = 2, Zrank = 2)
  for (x1,x2,z1,z2) in Iterators.product(0:1,0:1,0:1,0:1)
    for (X,Z) in Iterators.product(0:1,0:1)
      el = ((x1⊻x2⊻X== 0) && (z1⊻z2⊻Z == 0) ? pI :
            (x1⊻x2⊻X== 1) && (z1⊻z2⊻Z == 0) ? pX :
            (x1⊻x2⊻X== 0) && (z1⊻z2⊻Z == 1) ? pZ :
                                       pY)
      @test H[X+1,Z+1][x1+1,x2+1,z1+1,z2+1] ≈ el
    end
  end
end

