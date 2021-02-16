using Test
include("surfacecode.jl")
include("decoder.jl")

#@testset "surface code contructor" begin
#  d = 15
#  code = SurfaceCode(d)
#  @test d ≈ code.d
#  @test d ≈ distance(code)
#end
#
#@testset "qubit at coordinate" begin
#  d = 15
#  code = SurfaceCode(d)
#  # horizontal qubits
#  for y in 1:2:2*d-1
#    for x in 1:2:2*d-1
#      q = Q_at(x,y,d)
#      @test q == (2*d-1)*(y-1)÷2 + (x+1)÷2
#      r = code.Qcoord[q]
#      @test r[1] == x
#      @test r[2] == y
#    end
#  end
#  # horizontal qubits
#  for y in 2:2:2*d-1
#    for x in 2:2:2*d-1
#      q = Q_at(x,y,d)
#      @test q == (2*d-1)*(y-2)÷2 + x÷2 + d 
#      r = code.Qcoord[q] 
#      @test r[1] == x
#      @test r[2] == y
#    end
#  end
#end
#
#@testset "stabilizers at coordinate" begin
#  d = 15
#  code = SurfaceCode(d)
#  # X stabilizers 
#  c = 1
#  for y in 1:2:2*d-1
#    for x in 2:2:2*d-2
#      X = X_at(x,y,d)
#      @test X == c
#      r = code.Xcoord[X]
#      @test r[1] == x
#      @test r[2] == y
#      c+= 1
#    end
#  end
#  # X stabilizers 
#  c = 1
#  for y in 2:2:2*d-1
#    for x in 1:2:2*d-1
#      Z = Z_at(x,y,d)
#      @test Z == c
#      r = code.Zcoord[Z]
#      @test r[1] == x
#      @test r[2] == y
#      c+= 1
#    end
#  end
#end
#
#@testset "error chains" begin
#  d = 15
#  probs = (pX = 0.1, pY = 0.1, pZ = 0.1)
#  code = SurfaceCode(d)
#  
#  e = errorchain(10; error_probability = probs)
#  @test length(e) == 10
#  e = errorchain(code; error_probability = probs)
#  @test length(e) == nqubits(code)
#
#  nchains = 50
#  E = errorchains(code,nchains; error_probability = probs)
#  @test length(E) == nchains
#end
#
#@testset "syndromes" begin
#  d = 15
#  code = SurfaceCode(d)
#  N = nqubits(code)
#  probs = (pX = 0.1, pY = 0.1, pZ = 0.1) 
#  for i in 1:100
#    E = errorchain(code; error_probability = probs)
#    @test length(E) == N 
#    S = syndrome(E, code)
#    @test length(S[:X]) == d*(d-1)
#    # apply random stabilizers
#    for j in 1:10
#      stabX = code.QonS[:X][X_at(rand(2:2:2*(d-1)),rand(1:2:2*d-1),d)] 
#      stabZ = code.QonS[:Z][Z_at(rand(1:2:2*d-1),rand(2:2:2*(d-1)),d)]
#      pauli = combine_pauliXZ(topauli(N,stabX), topauli(N,stabZ))
#      Ẽ = E ⊙ pauli
#      S̃ = syndrome(Ẽ,code)
#      @test S == S̃
#    end 
#  end
#end
#@testset "reference pauli" begin
#  d = 15
#  nsamples = 100
#  code = SurfaceCode(d)
#  probs = (pX = 0.1, pY = 0.1, pZ = 0.1)
#  E, S = getsamples(code, nsamples; error_probability = probs)  
#
#  for n in 1:nsamples
#    f = referencepauli(S[n], code)
#    s̃ = syndrome(f, code)
#    @test S[n] == s̃
#    effective_pauli = E[n] ⊙ f
#    s_trivial = syndrome(effective_pauli, code)
#    @test all(s_trivial[:Z] .== 0)
#    @test all(s_trivial[:X] .== 0)
#  end
#end

#@testset "logical operator" begin
#  d = 5
#  code = SurfaceCode(d)
#
#  # check that logical operators do not trigger a syndrome
#  logicalI = logicaloperator(code, "I")
#  logicalX = logicaloperator(code, "X")
#  logicalZ = logicaloperator(code, "Z")
#  logicalY = logicaloperator(code, "Y")
#  
#  s = syndrome(logicalI, code)
#  @test all(s[:Z] .== 0)
#  @test all(s[:X] .== 0)
#  s = syndrome(logicalX, code)
#  @test all(s[:Z] .== 0)
#  @test all(s[:X] .== 0)
#  s = syndrome(logicalZ, code)
#  @test all(s[:Z] .== 0)
#  @test all(s[:X] .== 0)
#  s = syndrome(logicalY, code)
#  @test all(s[:Z] .== 0)
#  @test all(s[:X] .== 0)
#end
#
#
#@testset "Wilson loops" begin
#  d = 5
#  code = SurfaceCode(d)
#  probs = (pX = 0.1, pY = 0.1, pZ = 0.1)
#  nsamples = 100
#  E, S = getsamples(code, nsamples; error_probability = probs)
#  for n in 1:nsamples
#    e = E[n]
#    s = S[n]
#    f = referencepauli(s, code)
#    r = e ⊙ f
#    s̃ = syndrome(r, code)
#    @test all(s̃[:Z] .== 0)
#    @test all(s̃[:X] .== 0)
#    wX,wZ = Wilsonloops(r, code)
#
#    # apply logical X
#    logicalX = logicaloperator(code,"X")
#    L = r ⊙ logicalX
#    s̃ = syndrome(L, code)
#    @test all(s̃[:Z] .== 0)
#    @test all(s̃[:X] .== 0)
#    w̃X, w̃Z = Wilsonloops(L, code)
#    @test wX == w̃X ⊻ 1
#    @test wZ == w̃Z
#    
#    # apply logical X
#    logicalZ = logicaloperator(code,"Z")
#    L = r ⊙ logicalZ
#    s̃ = syndrome(L, code)
#    @test all(s̃[:Z] .== 0)
#    @test all(s̃[:X] .== 0)
#    w̃X, w̃Z = Wilsonloops(L, code)
#    @test wX == w̃X
#    @test wZ == w̃Z ⊻ 1
#  
#    # apply logical X
#    logicalY = logicaloperator(code,"Y")
#    L = r ⊙ logicalY
#    s̃ = syndrome(L, code)
#    @test all(s̃[:Z] .== 0)
#    @test all(s̃[:X] .== 0)
#    w̃X, w̃Z = Wilsonloops(L, code)
#    @test wX == w̃X ⊻ 1
#    @test wZ == w̃Z ⊻ 1
#  end
#
#end

@testset "v tensors" begin
  d = 15
  code = SurfaceCode(d)
  pX = 0.05
  pY = 0.01
  pZ = 0.15
  pI = 1 - pX - pY - pZ
  probs = (pX = pX, pY = pY, pZ = pZ)  
  
  #t = array(vtensor(x,z,q,q',l1,l2; error_probability = probs))
  V = vtensor(probs)
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
  H = htensor(1,1,probs)
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
  H = htensor(2,1,probs)
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
  H = htensor(1,2,probs)
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
  H = htensor(2,2,probs)
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


#@testset "configure" begin
#
#  d = 15
#  code = SurfaceCode(d)
#  pX = 0.05
#  pY = 0.01
#  pZ = 0.15
#  pI = 1 - pX - pY - pZ
#  probs = (pX = pX, pY = pY, pZ = pZ)  
#  L,B,R=configure(d; error_probability = probs) 
#  @test length(L) == 2*d-1
#  @test length(R) == 2*d-1
#  @test length(B) == 2*d-3
#  @test length.(B) == repeat([2*d-1],length(B))
#  
#  N = 2*d-1
#  for (X,Z) in Iterators.product(0:1,0:1)
#    for n in 1:N
#      isodd(n)  && (@test length(inds(L[n][X.+1,Z.+1], tags= "Qubit"))==1)
#      iseven(n) && (@test length(inds(L[n], tags= "Qubit"))==1)
#    end
#    @test length(inds(L[1][X.+1,Z.+1], tags= "Link"))==1
#    for n in 2:N-1
#      isodd(n)  && (@test length(inds(L[n][X.+1,Z.+1], tags= "Link"))==2)
#      iseven(n) && (@test length(inds(L[n], tags= "Link"))==2)
#    end
#    @test length(inds(L[N][X.+1,Z.+1], tags= "Link"))==1
#  end
#
#  for i in 2:2*d-2
#    U = B[i-1]
#    for (X,Z) in Iterators.product(0:1,0:1)
#      if iseven(i)
#        for n in 1:N
#          isodd(n) && (@test length(inds(U[n], tags= "Qubit"))==2)
#          isodd(n) && (@test length(inds(U[n], tags= "Qubit",plev=0))==1)
#          isodd(n) && (@test length(inds(U[n], tags= "Qubit",plev=1))==1)
#          iseven(n) && (@test length(inds(U[n][X.+1,Z.+1], tags= "Qubit"))==2)
#          iseven(n) && (@test length(inds(U[n][X.+1,Z.+1], tags= "Qubit",plev=0))==1)
#          iseven(n) && (@test length(inds(U[n][X.+1,Z.+1], tags= "Qubit",plev=1))==1)
#        end
#        @test length(inds(U[1], tags= "Link"))==1
#        for n in 2:N-1
#          isodd(n) && (@test length(inds(U[n], tags= "Link"))==2)
#          iseven(n) && (@test length(inds(U[n][X.+1,Z.+1],tags= "Link"))==2)
#        end
#        @test length(inds(U[N], tags= "Link"))==1
#      else
#        for n in 1:N
#          iseven(n) && (@test length(inds(U[n], tags= "Qubit"))==2)
#          iseven(n) && (@test length(inds(U[n], tags= "Qubit",plev=0))==1)
#          iseven(n) && (@test length(inds(U[n], tags= "Qubit",plev=1))==1)
#          isodd(n) && (@test length(inds(U[n][X.+1,Z.+1], tags= "Qubit"))==2)
#          isodd(n) && (@test length(inds(U[n][X.+1,Z.+1], tags= "Qubit",plev=0))==1)
#          isodd(n) && (@test length(inds(U[n][X.+1,Z.+1], tags= "Qubit",plev=1))==1)
#        end
#        @test length(inds(U[1][X.+1,Z.+1], tags= "Link"))==1
#        for n in 2:N-1
#          iseven(n) && (@test length(inds(U[n], tags= "Link"))==2)
#          isodd(n) && (@test length(inds(U[n][X.+1,Z.+1],tags= "Link"))==2)
#        end
#        @test length(inds(U[N][X.+1,Z.+1], tags= "Link"))==1
#      end
#    end
#  end
#
#  for (X,Z) in Iterators.product(0:1,0:1)
#    for n in 1:N
#      isodd(n)  && (@test length(inds(R[n][X.+1,Z.+1], tags= "Qubit"))==1)
#      iseven(n) && (@test length(inds(R[n], tags= "Qubit"))==1)
#    end
#    @test length(inds(L[1][X.+1,Z.+1], tags= "Link"))==1
#    for n in 2:N-1
#      isodd(n)  && (@test length(inds(R[n][X.+1,Z.+1], tags= "Link"))==2)
#      iseven(n) && (@test length(inds(R[n], tags= "Link"))==2)
#    end
#    @test length(inds(R[N][X.+1,Z.+1], tags= "Link"))==1
#  end
#end
#








@testset "Tensor selection" begin
  d = 5
  code = SurfaceCode(d)
  pX = 0.05
  pY = 0.01
  pZ = 0.15
  pI = 1 - pX - pY - pZ
  probs = (pX = pX, pY = pY, pZ = pZ)  
  E, S = getsamples(code, 10; error_probability = probs);
  L,B,R = configure(d; error_probability = probs) 
  for n in 1:1
    e = E[n]
    ΨL, Λ, ΨR = gettensors(e,L,B,R)
  




  end
end
