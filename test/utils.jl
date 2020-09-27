using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random

@testset "fullvector - native" begin
  psi = qubits(1)
  psi_vec = fullvector(psi, reverse = false)
  @test psi_vec ≈ [1., 0.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi, reverse = false)
  @test psi_vec ≈ [0., 1.]

  psi = qubits(2)
  psi_vec = fullvector(psi, reverse = false)
  @test psi_vec ≈ [1., 0., 0., 0.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi, reverse = false)
  @test psi_vec ≈ [0., 1., 0., 0.]
  applygate!(psi,"X",2)
  psi_vec = fullvector(psi, reverse = false)
  @test psi_vec ≈ [0., 0., 0., 1.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi, reverse = false)
  @test psi_vec ≈ [0., 0., 1., 0.]
end

@testset "fullvector - reverse" begin
  psi = qubits(1)
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [1., 0.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0., 1.]

  psi = qubits(2)
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [1., 0., 0., 0.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0., 0., 1., 0.]
  applygate!(psi,"X",2)
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0., 0., 0., 1.]
  applygate!(psi,"X",1)
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0., 1., 0., 0.]
end

@testset "fullmatrix for Itensor - reverse" begin
  psi = qubits(2)
  # control = 0, target = 0 -> 00 = 1 0 0 0
  applygate!(psi,"CX",(1,2))
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [1, 0, 0, 0]

  psi = qubits(2)
  # control = 0, target = 1 -> 01 = 0 1 0 0
  applygate!(psi,"X",2)
  applygate!(psi,"CX",(1,2))
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0, 1, 0, 0]

  psi = qubits(2)
  # control = 1, target = 0 -> 11 = 0 0 0 1
  applygate!(psi,"X",1)
  applygate!(psi,"CX",(1,2))
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0, 0, 0, 1]

  psi = qubits(2)
  # control = 1, target = 1 -> 10 = 0 0 1 0
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"CX",(1,2))
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0, 0, 1, 0]

  # INVERTED TARGET AND CONTROL
  psi = qubits(2)
  # target = 0, control = 0 -> 00 = 1 0 0 0
  applygate!(psi,"CX",(2,1))
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [1, 0, 0, 0]

  psi = qubits(2)
  # target = 0, control = 1 -> 11 = 0 0 0 1
  applygate!(psi,"X",2)
  applygate!(psi,"CX",(2,1))
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0, 0, 0, 1]

  psi = qubits(2)
  # target = 1, control = 0 -> 10 = 0 0 1 0
  applygate!(psi,"X",1)
  applygate!(psi,"CX",(2,1))
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0, 0, 1, 0]

  psi = qubits(2)
  # target = 1, control = 1 -> 01 = 0 1 0 0
  applygate!(psi,"X",1)
  applygate!(psi,"X",2)
  applygate!(psi,"CX",(2,1))
  psi_vec = fullvector(psi, reverse = true)
  @test psi_vec ≈ [0, 1, 0, 0]
end

#@testset "hilbert space" begin
#  N = 10
#  Ψ = randomstate(N)
#  ρ = randomstate(N;mixed=true)
#  U = randomstate(N)
#  Λ = randomstate(N;mixed=true)
#
#  h = hilbertspace(Ψ)
#  @test length(h) == N
#  for j in 1:length(N)
#    @test hastags(h[j],"Site")
#  end
#  h = hilbertspace(ρ)
#  @test length(h) == N
#  for j in 1:length(N)
#    @test hastags(h[j],"Site")
#  end
#  h = hilbertspace(U)
#  @test length(h) == N
#  for j in 1:length(N)
#    @test hastags(h[j],"Site")
#  end
#  h = hilbertspace(Λ)
#  @test length(h) == N
#  for j in 1:length(N)
#    @test hastags(h[j],"Site")
#  end
#end

@testset "replacehilbert: states" begin
  N = 3
  χ = 3
  printflag = false
  
  # MPS <- MPS
  Ψref = randomstate(N)
  Ψ = randomstate(N)
  Ψprod = prod(Ψ)

  measure_before = fidelity(Ψref,Ψ)
  replacehilbertspace!(Ψ,Ψref)
  measure_after = fidelity(Ψref,Ψ)
  
  @test array(Ψprod) ≈ array(prod(Ψ)) 
  @test measure_before ≈ measure_after
  for j in 1:N
    @test firstind(Ψ[j],tags="Site") == firstind(Ψref[j],tags="Site")
  end
  if printflag
    println("Initialize MPS with MPS")
    @show Ψref
    @show Ψ
  end
 

  # MPS <- regular MPO (state)
  Ψ = randomstate(N)
  ρref = randomstate(N;mixed=true)
  Ψprod = prod(Ψ)
  
  measure_before = fullmatrix(ρref) * fullvector(Ψ) 
  replacehilbertspace!(Ψ,ρref)
  measure_after = fullvector(ρref * Ψ)
  
  @test array(Ψprod) ≈ array(prod(Ψ)) 
  @test measure_before ≈ measure_after
  for j in 1:N
    @test firstind(Ψ[j],tags="Site") == firstind(ρref[j],tags="Site",plev=0)
  end
  if printflag
    println("Initialize MPS with MPO")
    @show ρref
    @show Ψ
  end
  
  # MPS <- purified MPO (state)
  Ψ = randomstate(N)
  Λref = randomstate(N;lpdo=true)
  Ψprod = prod(Ψ)
  
  measure_before = fullfidelity(Ψ,Λref)
  replacehilbertspace!(Ψ,Λref)
  measure_after = fullfidelity(Ψ,Λref)

  @test array(Ψprod) ≈ array(prod(Ψ)) 
  @test measure_before ≈ measure_after
  for j in 1:N
    @test firstind(Ψ[j],tags="Site") == firstind(Λref.X[j],tags="Site")
  end
  if printflag
    println("Initialize MPS with LPDO")
    @show Λref
    @show Ψ
  end

  
  # regular MPO <- MPS
  Ψref = randomstate(N)
  ρ = randomstate(N;mixed=true)
  ρ2 = copy(ρ)
  
  measure_before = fullmatrix(ρ) * fullvector(Ψref)
  replacehilbertspace!(ρ,Ψref)
  measure_after = fullvector(ρ * Ψref)
  @test measure_before ≈ measure_after
  for j in 1:N
    @test array(ρ[j]) ≈ array(ρ2[j])
    @test firstind(Ψref[j],tags="Site") == firstind(ρ[j],tags="Site",plev=0)
    @test firstind(Ψref[j],tags="Site")' == firstind(ρ[j],tags="Site",plev=1)
  end
  if printflag
    println("Initialize MPO with MPS")
    @show Ψref
    @show ρ 
  end
  
  
  # regular MPO <- regular MPO (state)
  ρ = randomstate(N;mixed=true)
  ρref = randomstate(N;mixed=true)
  ρ2 = copy(ρ)
  
  measure_before = tr(conj(transpose(fullmatrix(ρ))) * fullmatrix(ρref))
  replacehilbertspace!(ρ,ρref)
  measure_after = inner(ρ,ρref)#fullmatrix(ρ * ρref)
  
  @test measure_before ≈ measure_after atol=1e-8
  for j in 1:N
    @test array(ρ[j]) ≈ array(ρ2[j])
    @test firstind(ρref[j],tags="Site",plev=0) == firstind(ρ[j],tags="Site",plev=0)
    @test firstind(ρref[j],tags="Site",plev=1) == firstind(ρ[j],tags="Site",plev=1)
  end
  if printflag
    println("Initialize MPO with MPO")
    @show ρref
    @show ρ 
  end
  
  # regular MPO <- purified MPO (state)
  ρ = randomstate(N;mixed=true)
  Λref = randomstate(N;lpdo=true)
  ρ2 = copy(ρ)

  measure_before = tr(conj(transpose(fullmatrix(ρ))) * fullmatrix(Λref)) 
  replacehilbertspace!(ρ,Λref)
  measure_after = inner(ρ,Λref)
  @test measure_before ≈ measure_after
  for j in 1:N
    @test array(ρ[j]) ≈ array(ρ2[j])
    @test firstind(Λref.X[j],tags="Site",plev=0) == firstind(ρ[j],tags="Site",plev=0)
    @test firstind(Λref.X[j],tags="Site",plev=0)' == firstind(ρ[j],tags="Site",plev=1)
  end
  if printflag
    println("Initialize MPO with LPDO")
    @show Λref
    @show ρ 
  end

  # LPDO <- MPS
  Λ = randomstate(N;lpdo=true)
  Ψref = randomstate(N)
  
  measure_before = fullfidelity(Ψref,Λ)
  replacehilbertspace!(Λ,Ψref)
  measure_after = fidelity(Λ,Ψref)
  @test measure_before ≈ measure_after atol=1e-7
  for j in 1:N
    @test firstind(Ψref[j],tags="Site") == firstind(Λ.X[j],tags="Site",plev=0)
  end
  if printflag
    println("Initialize LPDO with MPS")
    @show Ψref
    @show Λ 
  end
  
  # LPDO <- regular MPO (state)
  Λ = randomstate(N;lpdo=true)
  ρref = randomstate(N;mixed=true)
  
  measure_before = tr(conj(transpose(fullmatrix(ρref))) * fullmatrix(Λ))
  replacehilbertspace!(Λ,ρref)
  measure_after = inner(ρref,Λ)
  @test measure_before ≈ measure_after atol=1e-7
  for j in 1:N
    @test firstind(ρref[j],tags="Site",plev=0) == firstind(Λ.X[j],tags="Site",plev=0)
    @test firstind(ρref[j],tags="Site",plev=1) == firstind(Λ.X[j],tags="Site",plev=0)'
  end
  if printflag
    println("Initialize LPDO with MPO")
    @show ρref
    @show Λ 
  end

  # LPDO <- purified MPO (state)
  Λ    = randomstate(N;lpdo=true)
  Λref = randomstate(N;lpdo=true)

  measure_before = fullfidelity(Λ,Λref)
  replacehilbertspace!(Λ,Λref)
  measure_after = fullfidelity(Λ,Λref)
  @test measure_before ≈ measure_after atol=1e-7
  for j in 1:N
    @test firstind(Λ.X[j],tags="Site",plev=0) == firstind(Λ.X[j],tags="Site",plev=0)
  end
  if printflag
    println("Initialize LPDO with LPDO")
    @show Λref
    @show Λ
  end
end


@testset "replacehilbert: channels" begin
  N = 3
  χ = 3
  printflag = false
  
  # regular MPO <- regular MPO
  ρ = randomprocess(N;mixed=false)
  ρref = randomprocess(N;mixed=false)
  replacehilbertspace!(ρ,ρref)
  for j in 1:N
    @test firstind(ρref[j],tags="Site",plev=0) == firstind(ρ[j],tags="Site",plev=0)
    @test firstind(ρref[j],tags="Site",plev=1) == firstind(ρ[j],tags="Site",plev=1)
  end
  if printflag
    println("Initialize MPO with MPO")
    @show ρref
    @show ρ
  end
  
  # regular MPO <- purified MPO
  ρ    = randomprocess(N;mixed=false) # Circuit MPO
  Λref = randomprocess(N;mixed=true)  # Choi matrix LPDO
  replacehilbertspace!(ρ,Λref)
  for j in 1:N
    @test firstind(Λref.X[j],tags="Input") == firstind(ρ[j],tags="Input")
    @test firstind(Λref.X[j],tags="Output")' == firstind(ρ[j],tags="Output")
  end
  if printflag
    println("Initialize MPO with LPDO")
    @show Λref
    @show ρ
  end

  
  # Purified MPO  <- regular MPO
  Λ = randomprocess(N;mixed=true)
  ρref = randomprocess(N;mixed=false)
  replacehilbertspace!(Λ,ρref)
  for j in 1:N
    @test firstind(ρref[j],tags="Input") == firstind(Λ.X[j],tags="Input")
    @test firstind(ρref[j],tags="Output") == firstind(Λ.X[j],tags="Output")'
  end
  if printflag
    println("Initialize MPO with MPO")
    @show ρref
    @show Λ
  end
  
  # Purified MPO  <- regular MPO
  Λ = randomprocess(N;mixed=true)
  Λref = randomprocess(N;mixed=true)
  replacehilbertspace!(Λ,Λref)
  for j in 1:N
    @test firstind(Λref.X[j],tags="Input") == firstind(Λ.X[j],tags="Input")
    @test firstind(Λref.X[j],tags="Output") == firstind(Λ.X[j],tags="Output")
  end
  if printflag
    println("Initialize MPO with MPO")
    @show Λref
    @show Λ
  end

end


