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

@testset "v tensors" begin
  d = 15
  code = SurfaceCode(d)
  pX = 0.05
  pY = 0.01
  pZ = 0.15
  pI = 1 - pX - pY - pZ
  probs = (pX = pX, pY = pY, pZ = pZ)  
  
  #t = array(vtensor(x,z,q,q',l1,l2; error_probability = probs))
  V = PastaQ.vtensor(probs)
  for (xU, xD, zL, zR) in Iterators.product(0:1,0:1,0:1,0:1)
    for (X,Z) in Iterators.product(0:1,0:1)
      el = ((xU⊻xD⊻X == 0) && (zL⊻zR⊻Z == 0) ? pI :
            (xU⊻xD⊻X == 1) && (zL⊻zR⊻Z == 0) ? pX :
            (xU⊻xD⊻X == 0) && (zL⊻zR⊻Z == 1) ? pZ :
                                       pY)
      @test V[X+1,Z+1,xU+1,xD+1,zL+1,zR+1] ≈ el
    end
  end
end
@testset "h tensors" begin
  d = 15
  code = SurfaceCode(d)
  pX = 0.05
  pY = 0.01
  pZ = 0.15
  pI = 1 - pX - pY - pZ
  probs = (pX = pX, pY = pY, pZ = pZ)  
  
  ## 2 legs (X / Z)
  H = PastaQ.htensor(1,1,probs)
  for (x,z) in Iterators.product(0:1,0:1)
    for (X,Z) in Iterators.product(0:1,0:1)
      el = ((x⊻X == 0) && (z⊻Z == 0) ? pI :
            (x⊻X == 1) && (z⊻Z == 0) ? pX :
            (x⊻X == 0) && (z⊻Z == 1) ? pZ :
                                       pY)
      @test H[X+1,Z+1,x+1,z+1] ≈ el
    end
  end
  
  # 3 legs (X / X / Z)
  H = PastaQ.htensor(2,1,probs)
  for (x1,x2,z) in Iterators.product(0:1,0:1,0:1)
    for (X,Z) in Iterators.product(0:1,0:1)
      el = ((x1⊻x2⊻X == 0) && (z⊻Z == 0) ? pI :
            (x1⊻x2⊻X == 1) && (z⊻Z == 0) ? pX :
            (x1⊻x2⊻X == 0) && (z⊻Z == 1) ? pZ :
                                       pY)
      @test H[X+1,Z+1,x1+1,x2+1,z+1] ≈ el
    end
  end

  # 3 legs (X / Z / Z)
  H = PastaQ.htensor(1,2,probs)
  for (x,z1,z2) in Iterators.product(0:1,0:1,0:1)
    for (X,Z) in Iterators.product(0:1,0:1)
      el = ((x⊻X == 0) && (z1⊻z2⊻Z == 0) ? pI :
            (x⊻X == 1) && (z1⊻z2⊻Z == 0) ? pX :
            (x⊻X == 0) && (z1⊻z2⊻Z == 1) ? pZ :
                                       pY)
      @test H[X+1,Z+1,x+1,z1+1,z2+1] ≈ el
    end
  end
  # 4 legs (X / X / Z / Z)
  H = PastaQ.htensor(2,2,probs)
  for (x1,x2,z1,z2) in Iterators.product(0:1,0:1,0:1,0:1)
    for (X,Z) in Iterators.product(0:1,0:1)
      el = ((x1⊻x2⊻X== 0) && (z1⊻z2⊻Z == 0) ? pI :
            (x1⊻x2⊻X== 1) && (z1⊻z2⊻Z == 0) ? pX :
            (x1⊻x2⊻X== 0) && (z1⊻z2⊻Z == 1) ? pZ :
                                       pY)
      @test H[X+1,Z+1,x1+1,x2+1,z1+1,z2+1] ≈ el
    end
  end
end


##@testset "configure" begin
##
##  d = 15
##  code = SurfaceCode(d)
##  pX = 0.05
##  pY = 0.01
##  pZ = 0.15
##  pI = 1 - pX - pY - pZ
##  probs = (pX = pX, pY = pY, pZ = pZ)  
##  L,B,R=configure(d; error_probability = probs) 
##  @test length(L) == 2*d-1
##  @test length(R) == 2*d-1
##  @test length(B) == 2*d-3
##  @test length.(B) == repeat([2*d-1],length(B))
##  
##  N = 2*d-1
##  for (X,Z) in Iterators.product(0:1,0:1)
##    for n in 1:N
##      isodd(n)  && (@test length(inds(L[n][X.+1,Z.+1], tags= "Qubit"))==1)
##      iseven(n) && (@test length(inds(L[n], tags= "Qubit"))==1)
##    end
##    @test length(inds(L[1][X.+1,Z.+1], tags= "Link"))==1
##    for n in 2:N-1
##      isodd(n)  && (@test length(inds(L[n][X.+1,Z.+1], tags= "Link"))==2)
##      iseven(n) && (@test length(inds(L[n], tags= "Link"))==2)
##    end
##    @test length(inds(L[N][X.+1,Z.+1], tags= "Link"))==1
##  end
##
##  for i in 2:2*d-2
##    U = B[i-1]
##    for (X,Z) in Iterators.product(0:1,0:1)
##      if iseven(i)
##        for n in 1:N
##          isodd(n) && (@test length(inds(U[n], tags= "Qubit"))==2)
##          isodd(n) && (@test length(inds(U[n], tags= "Qubit",plev=0))==1)
##          isodd(n) && (@test length(inds(U[n], tags= "Qubit",plev=1))==1)
##          iseven(n) && (@test length(inds(U[n][X.+1,Z.+1], tags= "Qubit"))==2)
##          iseven(n) && (@test length(inds(U[n][X.+1,Z.+1], tags= "Qubit",plev=0))==1)
##          iseven(n) && (@test length(inds(U[n][X.+1,Z.+1], tags= "Qubit",plev=1))==1)
##        end
##        @test length(inds(U[1], tags= "Link"))==1
##        for n in 2:N-1
##          isodd(n) && (@test length(inds(U[n], tags= "Link"))==2)
##          iseven(n) && (@test length(inds(U[n][X.+1,Z.+1],tags= "Link"))==2)
##        end
##        @test length(inds(U[N], tags= "Link"))==1
##      else
##        for n in 1:N
##          iseven(n) && (@test length(inds(U[n], tags= "Qubit"))==2)
##          iseven(n) && (@test length(inds(U[n], tags= "Qubit",plev=0))==1)
##          iseven(n) && (@test length(inds(U[n], tags= "Qubit",plev=1))==1)
##          isodd(n) && (@test length(inds(U[n][X.+1,Z.+1], tags= "Qubit"))==2)
##          isodd(n) && (@test length(inds(U[n][X.+1,Z.+1], tags= "Qubit",plev=0))==1)
##          isodd(n) && (@test length(inds(U[n][X.+1,Z.+1], tags= "Qubit",plev=1))==1)
##        end
##        @test length(inds(U[1][X.+1,Z.+1], tags= "Link"))==1
##        for n in 2:N-1
##          iseven(n) && (@test length(inds(U[n], tags= "Link"))==2)
##          isodd(n) && (@test length(inds(U[n][X.+1,Z.+1],tags= "Link"))==2)
##        end
##        @test length(inds(U[N][X.+1,Z.+1], tags= "Link"))==1
##      end
##    end
##  end
##
##  for (X,Z) in Iterators.product(0:1,0:1)
##    for n in 1:N
##      isodd(n)  && (@test length(inds(R[n][X.+1,Z.+1], tags= "Qubit"))==1)
##      iseven(n) && (@test length(inds(R[n], tags= "Qubit"))==1)
##    end
##    @test length(inds(L[1][X.+1,Z.+1], tags= "Link"))==1
##    for n in 2:N-1
##      isodd(n)  && (@test length(inds(R[n][X.+1,Z.+1], tags= "Link"))==2)
##      iseven(n) && (@test length(inds(R[n], tags= "Link"))==2)
##    end
##    @test length(inds(R[N][X.+1,Z.+1], tags= "Link"))==1
##  end
##end
#
#
#
#
##
##
##
##
##
##@testset "Tensor selection" begin
##  d = 5
##  code = SurfaceCode(d)
##  pX = 0.05
##  pY = 0.01
##  pZ = 0.15
##  pI = 1 - pX - pY - pZ
##  probs = (pX = pX, pY = pY, pZ = pZ)  
##  E, S = getsamples(code, 10; error_probability = probs);
##  L,B,R = configure(d; error_probability = probs) 
##  for n in 1:1
##    e = E[n]
##    ΨL, Λ, ΨR = gettensors(e,L,B,R)
##  
##
##
##
##
##  end
##end
