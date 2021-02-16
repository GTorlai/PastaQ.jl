using PastaQ
using ITensors
using Random
using LinearAlgebra
import StatsBase
using Printf
import ITensors: ⊙

"""
Addition mod 2
"""
function ⊙(A::Vector{<:Array{Int64}{1}},B::Vector{<:Array{Int64}{1}}) 
  length(A) ≠ length(B) && error("pauli have different lengths")
  return [A[j] .⊻ B[j] for j in 1:length(A)]
end

⊙(A::Vector{Int64}, B::Vector{Int64}) = A .⊻ B
⊙(A::Int, B::Int) = A ⊻ B

"""
Generage a `N`-qubit Pauli error with probabilities `pX`, `pY` and `pZ`.
"""
function errorchain(nqubits::Int64; error_probability::NamedTuple = (pX = 0.0, pY = 0.0, pZ = 0.0)) 
  pX, pY, pZ = error_probability[:pX], error_probability[:pY], error_probability[:pZ]
  pX+pY+pZ > 1.0 && error("total error rates greater than 1")
  return StatsBase.sample([[0,0], [1,0], [0,1], [1,1]], StatsBase.Weights([1-pX-pY-pZ, pX, pZ, pY]), nqubits)
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
  
  #Lx = topauli(nqubits(code), [y*(2*code.d-1)+2 for y in 0:code.d-1])
  Lx = topauli(nqubits(code), [y*(2*code.d-1)+1 for y in 0:code.d-1])
  logical == "X" && return combine_pauliXZ(Lx,zeros(Int64,nqubits(code)))
  
  #Lz = topauli(nqubits(code), [2*code.d-1+x for x in 1:code.d])
  Lz = topauli(nqubits(code), 1:code.d|>collect)
  logical == "Z" && return combine_pauliXZ(zeros(Int64,nqubits(code)), Lz)
  
  logical == "Y" && return combine_pauliXZ(Lx,Lz)
  
  error("Logical Pauli operator not recognized")
end

function logicaloperator(code::SurfaceCode, logical::Array{Int64})
  logical == [0,0] && return logicaloperator(code,"I")
  logical == [1,0] && return logicaloperator(code,"X")
  logical == [0,1] && return logicaloperator(code,"Z")
  logical == [1,1] && return logicaloperator(code,"Y")
  error("logical operator not recognized")
end


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
    Xsyndrome = Xsyndrome ⊙ [n in code.SonQ[q][:X] ? 1 : 0 for n in 1:length(Xsyndrome)] 
  end
  for q in Xerrored_qubits
    Zsyndrome = Zsyndrome ⊙ [n in code.SonQ[q][:Z] ? 1 : 0 for n in 1:length(Zsyndrome)] 
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
Measure the wilson loops
"""
function Wilsonloops(pauli::Vector{<:Array}, code::SurfaceCode)
  wX = sum(first.(pauli)[[2*code.d-1+x for x in 1:code.d]]) % 2
  wZ = sum(last.(pauli)[[y*(2*code.d-1)+2 for y in 0:code.d-1]]) % 2
  return [wX,wZ]
end

"""
Return the Pauli operator that moves a charge at a given location to the 
closest smooth boundary
"""
function movecharge(index::Int, code::SurfaceCode)
  x0,y0 = code.Xcoord[index]
  chargepath = (x0 > code.d-1 ? Q_at([(x+1,y0,code.d) for x in x0:2:2*(code.d-1)]) :
                                Q_at([(x-1,y0,code.d) for x in x0:-2:2]))
  return topauli(nqubits(code), chargepath)
end

"""
Return the Pauli operator that moves a flux at a given location to the 
closest rough boundary
"""
function moveflux(index::Int, code::SurfaceCode)
  x0,y0 = code.Zcoord[index]
  fluxpath = (y0 > code.d-1 ?  Q_at([(x0,y+1,code.d) for y in y0:2:2*(code.d-1)]) : 
                               Q_at([(x0,y-1,code.d) for y in y0:-2:2]))
  return topauli(nqubits(code), fluxpath)
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
      stabs = [bits[1] ⊙ bits[2], bits[3] ⊙ bits[4]]
      
      x = vcat(pauli,bits)
      T[(x.+1)...] = (pauli ⊙ stabs == [0,0] ? pI : 
                      pauli ⊙ stabs == [0,1] ? pZ :
                      pauli ⊙ stabs == [1,0] ? pX :
                      pauli ⊙ stabs == [1,1] ? pY :
                      error("something went wrong..."))
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
      
      stabs = ((nlegsX,nlegsZ) == (1,1) ? bits :
               (nlegsX,nlegsZ) == (2,1) ? [bits[1] ⊙ bits[2],  bits[3]] : 
               (nlegsX,nlegsZ) == (1,2) ? [bits[1],  bits[2] ⊙ bits[3]] :
               (nlegsX,nlegsZ) == (2,2) ? [bits[1] ⊙ bits[2],  bits[3] ⊙ bits[4]] :
               error("something went wrong..."))
      
      x = vcat(pauli,bits)
      T[(x.+1)...] = (pauli ⊙ stabs == [0,0] ? pI : 
                      pauli ⊙ stabs == [0,1] ? pZ :
                      pauli ⊙ stabs == [1,0] ? pX :
                      pauli ⊙ stabs == [1,1] ? pY :
                      error("something went wrong..."))
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
  # TODO: precompute itensors for all 20 configurations of the 5 tensors (4x5) 
  # only change boundary tensors when switching coset
  V      = vtensor(error_probability)
  H_XZ   = htensor(1,1,error_probability)
  H_XXZ  = htensor(2,1,error_probability)
  H_XZZ  = htensor(1,2,error_probability)
  H_XXZZ = htensor(2,2,error_probability)
  
  nthreads = Threads.nthreads()
  sites = siteinds("Qubit", N)
  linksΨ = [Index(2; tags="Link, l=$i") for i in 1:N-1]
  linksΛ  = [Index(2; tags="Link, l=$i") for i in 1:N-1]
  ϕ = Vector{MPS}()
  Ψ = Vector{MPS}()
  Λ = Vector{MPO}()
  for nthread in 1:nthreads
    M = ITensor[]; U = ITensor[];
    push!(M, ITensor(zeros(2,2), sites[1], linksΨ[1])) 
    push!(U, ITensor(zeros(2,2,2), sites[1], sites[1]', linksΛ[1])) 
    for j in 2:N-1
      push!(M, ITensor(zeros(2,2,2), linksΨ[j-1], sites[j], linksΨ[j]))
      push!(U, ITensor(zeros(2,2,2,2), linksΛ[j-1], sites[j], sites[j]', linksΛ[j]))
    end
    push!(M, ITensor(zeros(2,2), sites[N], linksΨ[N-1])) 
    push!(U, ITensor(zeros(2,2,2), sites[N], sites[N]', linksΛ[N-1])) 
    push!(ϕ, MPS(M))
    push!(Ψ, MPS(M))
    push!(Λ, MPO(U))
  end
  # loop over syndromes
  Threads.@threads for k in 1:length(S)
    s = S[k]
    nthread = Threads.threadid()
    #println(k)
    
    # reference pauli
    f = referencepauli(s, code)
    coset_logits = [] 
    
    # loop over the Z logical operator
    for cosetZ in [0,1] 
      pauli = f ⊙ logicaloperator(code, [0,cosetZ]) 
      
      #build right boundary
      links = linkinds(ϕ[nthread])
      locE = pauli[Q_at(N,1,d)] .+ 1
      ϕ[nthread][1] = ITensor(H_XZ[locE...,:,:], sites[1], links[1])
      for j in 2:N-1
        if isodd(j)
          locE = pauli[Q_at(N,j,d)] .+ 1
          ϕ[nthread][j] = ITensor(H_XZZ[locE...,:,:,:], sites[j], links[j-1],links[j])
        else
          ϕ[nthread][j] = δ(links[j-1], links[j], sites[j])
        end
      end
      locE = pauli[Q_at(N,N,d)] .+ 1
      ϕ[nthread][N] = ITensor(H_XZ[locE...,:,:], sites[N], links[N-1])
      
      Ψ[nthread] = copy(ϕ[nthread])
      
      # contract the bulk
      for i in reverse(2:N-1)
        links = linkinds(Λ[nthread])
        if iseven(i)
          Λ[nthread][1] = δ(links[1],sites[1],sites[1]')
          for j in 2:N-1
            if iseven(j)
              locE = pauli[Q_at(i,j,d)] .+ 1
              Λ[nthread][j] = ITensor(V[locE...,:,:,:,:], links[j-1],links[j],sites[j],sites[j]')
            else
              Λ[nthread][j] = δ(links[j-1], links[j], sites[j], sites[j]')
            end
          end
          Λ[nthread][N] = δ(links[N-1],sites[N],sites[N]')
        else
          locE = pauli[Q_at(i,1,d)] .+ 1
          Λ[nthread][1] = ITensor(H_XXZ[locE...,:,:,:], sites[1], sites[1]', links[1])
          for j in 2:N-1
            if isodd(j)
              #y = 2*d-1-(j-1)
              locE = pauli[Q_at(i,j,d)] .+ 1
              Λ[nthread][j] = ITensor(H_XXZZ[locE...,:,:,:,:],sites[j], sites[j]', links[j-1], links[j])
            else
              Λ[nthread][j] = δ(links[j-1], links[j], sites[j], sites[j]')
            end
          end
          locE = pauli[Q_at(i,N,d)] .+ 1
          Λ[nthread][N] = ITensor(H_XXZ[locE...,:,:,:], sites[N], sites[N]', links[N-1])
        end
        Ψ[nthread] = noprime(*(Λ[nthread], Ψ[nthread]; method = "naive", maxdim = 6))
      end
      
      for cosetX in [0,1]
        pauliL = pauli ⊙ logicaloperator(code, [cosetX,0])
        # Boundary MPS
        links = linkinds(ϕ[nthread]) 
        locE = pauliL[Q_at(1,1,d)] .+ 1
        ϕ[nthread][1] = ITensor(H_XZ[locE...,:,:], sites[1], links[1]) 
        for j in 2:N-1
          if isodd(j)
            locE = pauliL[Q_at(1,j,d)] .+ 1
            ϕ[nthread][j] = ITensor(H_XZZ[locE...,:,:,:], sites[j], links[j-1],links[j])
          else
            ϕ[nthread][j] = δ(links[j-1], links[j], sites[j])
          end
        end
        locE = pauliL[Q_at(1,N,d)] .+ 1
        ϕ[nthread][N] = ITensor(H_XZ[locE...,:,:], sites[N], links[N-1])

        coset_logit = inner(Ψ[nthread],ϕ[nthread])
        push!(coset_logits, coset_logit)
      end
    end
    
    ml_coset = argmax(coset_logits)
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
d = 5

code = SurfaceCode(d)
#pX = 0.113
#pY = 0.00
#pZ = 0.00
#probs = (pX = pX, pY = pY, pZ = pZ)
p = 0.19
probs = (pX = p/3, pY = p/3, pZ = p/3)
nsamples = 100

t = @elapsed begin
  E, S = getsamples(code, nsamples; error_probability = probs)
  recoveries = decode(S,code;error_probability = probs)  
end
@printf("Total time = %.3f sec\n",t)
@printf("Logical failure rate: %.3f ",failure_rate(recoveries,E,code))


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

