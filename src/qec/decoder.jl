using PastaQ
using ITensors
using Random
using LinearAlgebra
import StatsBase
using Printf
import ITensors: ⊙
"""
Generate error chains
"""
function errorchain(nqubits::Int64; error_probability::NamedTuple = (pX = 0.0, pY = 0.0, pZ = 0.0)) 
  pX, pY, pZ = error_probability[:pX], error_probability[:pY], error_probability[:pZ]
  pX+pY+pZ > 1.0 && error("total error rates greater than 1")
  return StatsBase.sample([[0,0],[1,0],[0,1],[1,1]], StatsBase.Weights([1-pX-pY-pZ, pX, pZ, pY]), nqubits)
end

errorchain(code::SurfaceCode; kwargs...) = 
  errorchain(nqubits(code); kwargs...)

errorchains(nqubits::Int64, nchains::Int64; kwargs...) = 
  [errorchain(nqubits; kwargs...) for j in 1:nchains] 

errorchains(code::SurfaceCode, nchains::Int; kwargs...) = 
  errorchains(nqubits(code), nchains; kwargs...)


"""
Generate a pauli chain from a qubit support:
[1,3,5] -> [1,0,1,0,1,...]
"""
topauli(nqubits::Int64, support::Vector{Int64}) = 
  [n in support ? 1 : 0 for n in 1:nqubits]

topauli(code::SurfaceCode, support::Vector{Int64}; kwargs...) = 
  topauli(nqubits(code), support; kwargs...)

"""
Generate a pauli XZ from X and Z supports
"""
function combine_pauliXZ(pauliX::Vector{Int64}, pauliZ::Vector{Int64}) 
  length(pauliX) ≠ length(pauliZ) && error("pauli have different lengths")
  return [[pauliX[j],pauliZ[j]] for j in 1:length(pauliX)]
end

combine_pauliXZ(code::SurfaceCode; kwargs...) = 
  combineXZ(nqubits(code); kwargs...)

"""
Generate a logical operator for the Surface code
"""
function logicaloperator(code::SurfaceCode, logical::String)
  logical == "I" && return [[0,0] for _ in 1:nqubits(code)] 
  Lx = [n in [y*(2*code.d-1)+2 for y in 0:code.d-1] ? 1 : 0 for n in 1:nqubits(code)]
  Lx = topauli(nqubits(code), [y*(2*code.d-1)+2 for y in 0:code.d-1])
  logical == "X" && return combine_pauliXZ(Lx,zeros(Int64,nqubits(code)))
  Lz = topauli(nqubits(code), [2*code.d-1+x for x in 1:code.d])
  logical == "Z" && return combine_pauliXZ(zeros(Int64,nqubits(code)), Lz)
  logical == "Y" && return combine_pauliXZ(Lx,Lz)
  error("Logical Pauli operator not recognized")
end


"""
Measure the wilson loops
"""
function Wilsonloops(pauli::Vector{<:Array}, code::SurfaceCode)
  wX = sum(first.(pauli)[[2*code.d-1+x for x in 1:code.d]]) % 2
  wZ = sum(last.(pauli)[[y*(2*code.d-1)+2 for y in 0:code.d-1]]) % 2
  return [wX,wZ]
end

"""
Combine two pauli operators
"""
function ⊙(A::Vector{<:Array{Int64}{1}},B::Vector{<:Array{Int64}{1}}) 
  length(A) ≠ length(B) && error("pauli have different lengths")
  return [A[j] .⊻ B[j] for j in 1:length(A)]
end

⊙(A::Vector{Int64}, B::Vector{Int64}) = A .⊻ B

"""
Extract syndrome
"""
function syndrome(error::Vector{<:Array{Int64}{1}}, code::SurfaceCode)
  Xsyndrome = zeros(Int64,length(code.Xcoord))
  Zsyndrome = zeros(Int64,length(code.Zcoord))
  
  Xerrored_qubits = findall(x -> x==1, first.(error))
  Zerrored_qubits = findall(x -> x==1, last.(error))
  
  # trigger the stabilizers
  for q in Zerrored_qubits
    Xsyndrome .⊻= [n in code.SonQ[q][:X] ? 1 : 0 for n in 1:length(Xsyndrome)] 
  end
  for q in Xerrored_qubits
    Zsyndrome .⊻= [n in code.SonQ[q][:Z] ? 1 : 0 for n in 1:length(Zsyndrome)] 
  end
  return (X = Xsyndrome, Z = Zsyndrome)
end

"""
Generate error/syndrome data
"""
function getsamples(code::SurfaceCode, nsamples::Int64; kwargs...)
  errors = errorchains(code, nsamples; kwargs...)
  syndromes = []
  for n in 1:nsamples
    s = syndrome(errors[n], code)
    push!(syndromes, s) 
  end
  return errors, syndromes
end

"""
Return the Pauli operator that moves a charge at a given location to the 
closest smooth boundary
"""
function movecharge(index::Int, code::SurfaceCode)
  pauli = zeros(Int,nqubits(code))
  x0,y0 = code.Xcoord[index]
  chargeline = (x0 > code.d-1 ? Q_at([(x+1,y0,code.d) for x in x0:2:2*(code.d-1)]) :
                                Q_at([(x-1,y0,code.d) for x in x0:-2:2]))
  pauli[chargeline] .=1
  return pauli
end

"""
Return the Pauli operator that moves a flux at a given location to the 
closest rough boundary
"""
function moveflux(index::Int, code::SurfaceCode)
  pauli = zeros(Int,nqubits(code))
  x0,y0 = code.Zcoord[index]
  fluxline = (y0 > code.d-1 ?  Q_at([(x0,y+1,code.d) for y in y0:2:2*(code.d-1)]) : 
                               Q_at([(x0,y-1,code.d) for y in y0:-2:2]))
  pauli[fluxline] .=1
  return pauli
end

"""
Generate a Pauli operator consistent with a given syndrome
"""
function referencepauli(s::NamedTuple, code::SurfaceCode)
  N = nqubits(code)
  pauliX= zeros(Int64,N) 
  pauliZ = zeros(Int64,N) 
  
  charges = findall(x -> x == 1, s[:X])
  fluxes  = findall(x -> x == 1, s[:Z])
  for charge in charges
    pauliZ = pauliZ ⊙ movecharge(charge, code)
  end
  for flux in fluxes
    pauliX = pauliX ⊙ moveflux(flux, code)
  end
  return combine_pauliXZ(pauliX, pauliZ)
end


function vtensor(error_probability::NamedTuple) 
  pX, pY, pZ = error_probability[:pX], error_probability[:pY], error_probability[:pZ]
  pI = 1.0 - pX - pY - pZ
  paulis = [[0,0],[1,0],[0,1],[1,1]]
  
  T = zeros(Float64,(2,2,2,2,2,2)) 
  for pauli in paulis
    for i in 0:1<<4-1
      bits = digits(i, base=2, pad=4) |> reverse
      x = vcat(pauli,bits)
      stabs = [bits[1] ⊻ bits[2], bits[3] ⊻ bits[4]]
      if pauli .⊻ stabs == [0,0]
        T[(x.+1)...] = pI
      elseif pauli .⊻ stabs == [0,1] 
        T[(x.+1)...] = pZ
      elseif pauli .⊻ stabs == [1,0]
        T[(x.+1)...] = pX
      elseif pauli .⊻ stabs == [1,1]
        T[(x.+1)...] = pY
      else
        error("something went wrong...")
      end
    end
  end
  return T
end

function htensor(nlegsX::Int64, nlegsZ::Int64, error_probability::NamedTuple)
  pX, pY, pZ = error_probability[:pX], error_probability[:pY], error_probability[:pZ]
  numlegs = nlegsX+nlegsZ
  pI = 1.0 - pX - pY - pZ
  paulis = [[0,0],[1,0],[0,1],[1,1]]
  T = zeros(Float64, repeat([2],numlegs+2)...)
  for pauli in paulis
    for i in 0:1<<numlegs-1
      bits = digits(i, base=2, pad=numlegs) |> reverse
      x = vcat(pauli,bits)
      if nlegsX == 1 && nlegsZ == 1
        stabs = bits
      elseif (nlegsX == 2) && (nlegsZ == 1)
        stabs = [bits[1] ⊻ bits[2], bits[3]]
      elseif nlegsX == 1 && nlegsZ == 2
        stabs = [bits[1], bits[2] ⊻ bits[3]]
      elseif nlegsX == 2 && nlegsZ == 2
        stabs = [bits[1] ⊻ bits[2], bits[3] ⊻ bits[4]]
      else
        error("something went wrong...")
      end
      if pauli .⊻ stabs == [0,0]
        T[(x.+1)...] = pI
      elseif pauli .⊻ stabs == [0,1] 
        T[(x.+1)...] = pZ
      elseif pauli .⊻ stabs == [1,0]
        T[(x.+1)...] = pX
      elseif pauli .⊻ stabs == [1,1]
        T[(x.+1)...] = pY
      else
        error("something went wrong...")
      end
    end
  end
  return T
end


function decode(S::Vector, code::SurfaceCode; error_probability::NamedTuple)
  nthreads = Threads.nthreads()
  d = code.d
  N = 2*d-1
  cosets = ["I","X","Z","Y"]
  recoveries = [[] for _ in 1:nthreads] 
  # Pre-compute each dense tensor
  V      = vtensor(error_probability)
  H_XZ   = htensor(1,1,error_probability)
  H_XXZ  = htensor(2,1,error_probability)
  H_XZZ  = htensor(1,2,error_probability)
  H_XXZZ = htensor(2,2,error_probability)
  
  #nthreads = Threads.nthreads()

  # loop over syndromes
  Threads.@threads for k in 1:length(S)
    s = S[k]
    nthread = Threads.threadid()
    
    # reference pauli
    f = referencepauli(s, code)
    
    coset_logits = [] 
    
    # loop over cosets
    for coset in cosets 
      s = siteinds("Qubit", N)
      pauli = f ⊙ logicaloperator(code, coset) 
      
      # Boundary MPS
      M = ITensor[]
      l = [Index(2; tags="Link, l=$i") for i in 1:N-1]
      
      locE = pauli[Q_at(1,1,d)] .+ 1
      push!(M, ITensor(H_XZ[locE...,:,:,:], s[1], l[1])) 
      for j in 2:N-1
        if isodd(j)
          y = 2*d-1-(j-1); 
          locE = pauli[Q_at(1,y,d)] .+ 1
          push!(M, ITensor(H_XZZ[locE...,:,:,:], s[j], l[j-1],l[j]))
        else
          push!(M, δ(l[j-1], l[j], s[j]))
        end
      end
      locE = pauli[Q_at(1,N,d)] .+ 1
      push!(M, ITensor(H_XZ[locE...,:,:,:], s[N], l[N-1]))
      Ψ = MPS(M) 
      
      for i in 2:N-1
        l = [Index(2; tags="Link, l=$i") for i in 1:N-1]
        M = ITensor[]
        if iseven(i)
          push!(M, δ(l[1],s[1],s[1]'))
          for j in 2:N-1
            if iseven(j)
              y = 2*d-1-(j-1)
              locE = pauli[Q_at(i,y,d)] .+ 1
              push!(M, ITensor(V[locE...,:,:,:,:], l[j-1],l[j],s[j],s[j]'))
            else
              push!(M, δ(l[j-1], l[j], s[j], s[j]'))
            end
          end
          push!(M, δ(l[N-1],s[N],s[N]'))
        else
          locE = pauli[Q_at(i,1,d)] .+ 1
          push!(M, ITensor(H_XXZ[locE...,:,:,:], s[1], s[1]', l[1]))
          for j in 2:N-1
            if isodd(j)
              y = 2*d-1-(j-1)
              locE = pauli[Q_at(i,y,d)] .+ 1
              push!(M, ITensor(H_XXZZ[locE...,:,:,:,:],s[j], s[j]', l[j-1], l[j]))
            else
              push!(M, δ(l[j-1], l[j], s[j], s[j]'))
            end
          end
          locE = pauli[Q_at(i,N,d)] .+ 1
          push!(M, ITensor(H_XXZ[locE...,:,:,:], s[N], s[N]', l[N-1]))
        end
        Λ = MPO(M)
        Ψ = noprime(*(Λ, Ψ; method = "naive", maxdim = 6))
      end
      
      M = ITensor[]
      l = [Index(2; tags="Link, l=$i") for i in 1:N-1]
      
      locE = pauli[Q_at(N,1,d)] .+ 1
      push!(M, ITensor(H_XZ[locE...,:,:,:], s[1], l[1])) 
      for j in 2:N-1
        if isodd(j)
          y = 2*d-1-(j-1); 
          locE = pauli[Q_at(N,y,d)] .+ 1
          push!(M, ITensor(H_XZZ[locE...,:,:,:], s[j], l[j-1],l[j]))
        else
          push!(M, δ(l[j-1], l[j], s[j]))
        end
      end
      locE = pauli[Q_at(N,N,d)] .+ 1
      push!(M, ITensor(H_XZ[locE...,:,:,:], s[N], l[N-1]))
      Φ = MPS(M) 
      
      #coset_probability =  inner(Ψ,Φ)
      #push!(coset_probabilities, coset_probability)
      coset_logit =  inner(Ψ,Φ)
      push!(coset_logits, coset_logit)
    end
    #@show nthread,coset_probabilities
    ml_coset = argmax(coset_logits)
    #ml_coset = argmax(coset_probabilities)
    logical_op = logicaloperator(code, cosets[ml_coset])
    recovery = f ⊙ logical_op
    push!(recoveries[nthread], recovery)
  end
  return vcat(recoveries...) 
end

function failure_rate(R::Vector, E::Vector, code::SurfaceCode)
  f_rate = 0.0
  ndata = length(E)
  @assert length(E) == length(R)
  
  for i in 1:length(E)
    net_pauli = E[i] ⊙ R[i] 
    wX,wZ = Wilsonloops(net_pauli, code)
    (wX == 1 || wZ == 1) && (f_rate += 1/ndata) 
  end
  return f_rate
end



Random.seed!(1234)
d = 15
code = SurfaceCode(d)
pX = 0.00
pY = 0.00
pZ = 0.10
probs = (pX = pX, pY = pY, pZ = pZ)
nsamples = 100
#E, S = getsamples(code, nsamples; error_probability = probs)

t = @elapsed begin
  E, S = getsamples(code, nsamples; error_probability = probs)
  recoveries = decode(S,code;error_probability = probs)  
end
@printf("Total time = %.3f sec",t)
#@show logical_error_rate



#nsamples = 1000
#p_list = [0.02,0.04,0.06,0.08,0.10,0.12,0.14]
#
#for p in p_list
#  local probs = (pX = 0.0, pY = 0.0, pZ = p)
#  local E, S = getsamples(code, nsamples; error_probability = probs)
#  local recoveries = decode(S,code;error_probability = probs)
#  local logical_error_rate = failure_rate(recoveries,E,code)
#  @printf("p = %.2f  :  Logical failure rate = %.3f\n",p,logical_error_rate)
#end
























#function configure(d::Int64; error_probability::NamedTuple = (pX = 0.0, pY = 0.0, pZ = 0.0))
#  #println("Initializing the tensors...")
#  N = 2*d-1 
#  s = siteinds("Qubit", N)
#  l = [[Index(2; tags="Link, l=$l") for l in 1:N-1] for _ in 1:N]
#  
#  # left MPS boundary
#  left = []
#  push!(left, htensor(l[1][1], s[1]; error_probability = error_probability))
#  for n in 2:N-1
#    if iseven(n)
#      push!(left, δ(l[1][n-1], l[1][n], s[n]))
#    else
#      #q = Q_at(1,y,d)
#      push!(left, htensor(l[1][n-1], l[1][n], s[n]; error_probability = error_probability))
#    end
#  end
#  push!(left, htensor(l[1][N-1], s[N]; error_probability = error_probability))
#  
#  bulk = []
#  for i in 2:N-1
#    layer = []
#    if iseven(i)
#      push!(layer, δ(s[1],s[1]',l[i][1]))
#      for n in 2:N-1
#        if iseven(n)
#          push!(layer, vtensor(s[n], s[n]', l[i][n-1], l[i][n]; error_probability = error_probability))
#        else
#          push!(layer, δ(s[n],s[n]',l[i][n-1], l[i][n]))
#        end
#      end
#      push!(layer, δ(s[N],s[N]',l[i][N-1]))
#    else
#      push!(layer, htensor(l[i][1], s[1], s[1]'; error_probability = error_probability))
#      for n in 2:N-1
#        if iseven(n)
#          push!(layer, δ(l[i][n-1], l[i][n], s[n], s[n]'))
#        else
#          push!(layer, htensor(l[i][n-1], l[i][n], s[n], s[n]'; error_probability = error_probability))
#        end
#      end
#      push!(layer, htensor(l[i][N-1], s[N], s[N]'; error_probability = error_probability))
#    end
#    push!(bulk, layer)
#  end
#  
#  right = []
#  push!(right, htensor(l[N][1], s[1]; error_probability = error_probability))
#  for n in 2:N-1
#    if iseven(n)
#      push!(right, δ(l[N][n-1], l[N][n], s[n]))
#    else
#      push!(right, htensor(l[N][n-1], l[N][n], s[n]; error_probability = error_probability))
#    end
#  end
#  push!(right, htensor(l[N][N-1], s[N]; error_probability = error_probability))
#  return left, bulk, right
#end
#
#
#function gettensors(f::Vector{<:Array}, L::Vector, B::Vector, R::Vector)
#  #println("Setting up tensor given the reference pauli...")
#  N = length(L)
#  d = (N+1)÷2 
#  # Select the tensor elements
#  M = ITensor[]
#  for j in 1:N
#    if isodd(j)
#      y = 2*d-1-(j-1)
#      localerror = f[Q_at(1,y,d)] .+ 1
#      push!(M, L[j][localerror...])
#    else
#      push!(M, L[j])
#    end
#  end
#  ΨL = MPS(M) 
#  Λ = []
#  for i in 2:N-1
#    M = ITensor[]
#    if iseven(i)
#      for j in 1:N
#        if iseven(j)
#          y = 2*d-1-(j-1)
#          localerror = f[Q_at(i,y,d)] .+ 1
#          push!(M, B[i-1][j][localerror...])
#        else
#          push!(M, B[i-1][j])
#        end
#      end
#    else
#      for j in 1:N
#        if isodd(j)
#          y = 2*d-1-(j-1)
#          localerror = f[Q_at(i,y,d)] .+ 1
#          push!(M, B[i-1][j][localerror...])
#        else
#          push!(M, B[i-1][j])
#        end
#      end
#    end
#    push!(Λ, MPO(M))
#  end
#  
#  M = ITensor[]
#  for j in 1:N
#    if isodd(j)
#      y = 2*d-1-(j-1)
#      localerror = f[Q_at(N,y,d)] .+ 1
#      push!(M, R[j][localerror...])
#    else
#      push!(M, R[j])
#    end
#  end
#  ΨR = MPS(M) 
#  return ΨL, Λ, ΨR
#end
#

