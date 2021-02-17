"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                               NOISE MODELS                                   -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

"""
    paulierror(nqubits::Int64; 
               error_probability::NamedTuple = (pX = 0.0, pY = 0.0, pZ = 0.0))


Generage a `N`-qubit Pauli error with probabilities `pX`, `pY` and `pZ`.
"""
function paulierror(nqubits::Int64; 
                    error_probability::NamedTuple = (pX = 0.0, pY = 0.0, pZ = 0.0))

  pX, pY, pZ = error_probability[:pX], error_probability[:pY], error_probability[:pZ]
  pX+pY+pZ > 1.0 && error("total error rates greater than 1")
  return StatsBase.sample([(0,0), (1,0), (0,1), (1,1)], StatsBase.Weights([1-pX-pY-pZ, pX, pZ, pY]), nqubits)
end

paulierror(nqubits::Int64, nsamples::Int64; kwargs...) = 
  [paulierror(nqubits; kwargs...) for j in 1:nsamples] 

paulierror(code::QuantumCode; kwargs...) = 
  paulierror(nqubits(code); kwargs...)

paulierror(code::QuantumCode, nsamples::Int; kwargs...) = 
  paulierror(nqubits(code), nsamples; kwargs...)

"""
    bitfliperror(nqubits::Int64; p::Float64) = 

Generate a `N`-qubit pauli X error with probability `p`.
"""
bitfliperror(nqubits::Int64; p::Float64) = 
  paulierror(nqubits; error_probability = (pX = p, pY = 0.0, pZ = 0.0))

bitfliperror(nqubits::Int64, nsamples::Int64; kwargs...) = 
  [bitfliperror(nqubits; kwargs...) for j in 1:nsamples]

bitfliperror(code::QuantumCode; kwargs...) = 
  bitfliperror(nqubits(code); kwargs...)

bitfliperror(code::QuantumCode, nsamples::Int64; kwargs...) = 
  bitfliperror(nqubits(code), nsamples; kwargs...)

"""
    phasefliperror(nqubits::Int64; p::Float64) = 

Generate a `N`-qubit pauli Z error with probability `p`.
"""
phasefliperror(nqubits::Int64; p::Float64) = 
  paulierror(nqubits; error_probability = (pX = 0.0, pY = 0.0, pZ = p))

phasefliperror(nqubits::Int64, nsamples::Int64; kwargs...) = 
  [phasefliperror(nqubits; kwargs...) for j in 1:nsamples]

phasefliperror(code::QuantumCode; kwargs...) = 
  phasefliperror(nqubits(code); kwargs...)

phasefliperror(code::QuantumCode, nsamples::Int64; kwargs...) = 
  phasefliperror(nqubits(code), nsamples; kwargs...)


"""
    depolarizingerror(nqubits::Int64; p::Float64) = 

Generate a `N`-qubit error with each Pauli having equal
probability `p/3`.
"""
depolarizingerror(nqubits::Int64; p::Float64) = 
  paulierror(nqubits; error_probability = (pX = p/3, pY = p/3, pZ = p/3))

depolarizingerror(nqubits::Int64, nsamples::Int64; kwargs...) = 
  [depolarizingerror(nqubits; kwargs...) for j in 1:nsamples]

depolarizingerror(code::QuantumCode; kwargs...) = 
  depolarizingerror(nqubits(code); kwargs...)

depolarizingerror(code::QuantumCode, nsamples::Int64; kwargs...) = 
  depolarizingerror(nqubits(code), nsamples; kwargs...)

"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                            TENSOR NETWORK DECODER                            -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

"""
    syndrome(code::QuantumCode, error::Vector{<:Vector{Tuple{Int64,Int64}}})

Return the syndromes of a set of errors for a given input code.
Each error is provided as a Vector of tuples:

E = [(X₁,Z₁),(X₂,X₂),...,]
"""
function syndrome(error::Vector{<:Vector{Tuple{Int64,Int64}}}, code::QuantumCode)
  
  syndromes = Vector{NamedTuple}()
  
  # get all the stabilizers connected to each qubit
  stabZ_at = stabZ_atQubit(code)
  stabX_at = stabX_atQubit(code)
  
  # loop over each error
  for e in error
    # find the location of Z and X errors
    Zerrored_qubits = findall(x -> x==1, last.(e))
    Xerrored_qubits = findall(x -> x==1, first.(e))
    
    # initialize empty syndrome vectors
    Zsyndrome = zeros(Int64,length(coordinates_ofStabZ(code)))
    Xsyndrome = zeros(Int64,length(coordinates_ofStabX(code)))
    
    # trigger the X stabilizers with Z errors
    for qubit in Xerrored_qubits
      Zsyndrome = Zsyndrome ⊙ [n in stabZ_at[qubit] ? 1 : 0 for n in 1:length(Zsyndrome)] 
    end
    # trigger the Z  stabilizers with X errors
    for qubit in Zerrored_qubits
      Xsyndrome = Xsyndrome ⊙ [n in stabX_at[qubit] ? 1 : 0 for n in 1:length(Xsyndrome)] 
    end
    
    push!(syndromes, (X = Xsyndrome, Z = Zsyndrome))
  end
  return syndromes 
end

syndrome(error::Vector{Tuple{Int64,Int64}}, code::QuantumCode) = 
  syndrome([error], code)[1]


"""
    codetensor(nlegX::Int64, nlegsZ::Int64, error_probability::NamedTuple)

Generate a tensor in the TN for the QEC code.
"""
function codetensor(error_probability::NamedTuple;
                    Xrank::Int64 = 1, Zrank::Int64 = 1)
  # single-qubit pauli error rates
  pX, pY, pZ = error_probability[:pX], error_probability[:pY], error_probability[:pZ]
  
  rank = Xrank + Zrank
  pI = 1.0 - pX - pY - pZ
  paulis = [[0,0],[1,0],[0,1],[1,1]]
  T = zeros(Float64, repeat([2], rank + 2)...)
  for pauli in paulis
    for i in 0:1 << rank-1
      bits = digits(i, base = 2, pad = rank) |> reverse
      
      stabs = ((Xrank, Zrank) == (1,1) ? bits :
               (Xrank, Zrank) == (2,1) ? [bits[1] ⊙ bits[2],  bits[3]] : 
               (Xrank, Zrank) == (1,2) ? [bits[1],  bits[2] ⊙ bits[3]] :
               (Xrank, Zrank) == (2,2) ? [bits[1] ⊙ bits[2],  bits[3] ⊙ bits[4]] :
               error("something went wrong..."))
      
      tensor_el = vcat(pauli,bits)
      T[(tensor_el .+ 1)...] = (pauli ⊙ stabs == [0,0] ? pI : 
                                pauli ⊙ stabs == [0,1] ? pZ :
                                pauli ⊙ stabs == [1,0] ? pX :
                                pauli ⊙ stabs == [1,1] ? pY :
                                error("something went wrong..."))
    end
  end
  return T
end

"""
    decode(S::Vector, code::SurfaceCode; error_probability::NamedTuple)

Determine the most likely errors corresponding to a set of input syndromes,
and returns the appropriate recovery operation.
"""
function decode(S::Vector, code::SurfaceCode; 
                error_probability::NamedTuple,
                maxdim::Int64 = 10_000,
                contract_method::String = "naive")

  nthreads = Threads.nthreads()
  d = distance(code)
  N = 2*d-1
  cosets = ["I","X","Z","Y"]
  recoveries = [[] for _ in 1:nthreads] 
  
  # Pre-compute each dense tensor
  # TODO: precompute itensors for all 20 configurations of the 5 tensors (4x5) 
  V      = codetensor(error_probability; Xrank = 2, Zrank = 2)
  H_XZ   = codetensor(error_probability; Xrank = 1, Zrank = 1)
  H_XXZ  = codetensor(error_probability; Xrank = 2, Zrank = 1)
  H_XZZ  = codetensor(error_probability; Xrank = 1, Zrank = 2)
  H_XXZZ = codetensor(error_probability; Xrank = 2, Zrank = 2)
  
  # generate set of tensor indices
  sites   = siteinds("Qubit", N)
  linksΨ  = [Index(2; tags="Link, l=$i") for i in 1:N-1]
  linksΛ  = [Index(2; tags="Link, l=$i") for i in 1:N-1]
  
  # pre-allocate memory
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
  
  # hyper-thread the syndrome loop
  Threads.@threads for k in 1:length(S)
    s = S[k]
    nthread = Threads.threadid()
    
    # generate a pure pauli error 
    f = purepaulierror(s, code)
    
    coset_probabilities = [] 
    # loop over the Z logical operator
    for cosetZ in [0,1] 
      pauli = f ⊙ logicaloperator([0,cosetZ], code) 
      
      # build right boundary MPS
      links = linkinds(ϕ[nthread])
      
      ϕ[nthread][1] = ITensor(H_XZ[(pauli[qubit_at(N, 1, d)].+1)...,:,:], 
                              sites[1], links[1])
      for j in 2:N-1
        isodd(j) &&  (ϕ[nthread][j] = ITensor(H_XZZ[(pauli[qubit_at(N,j,d)].+1)...,:,:,:],
                                              sites[j], links[j-1], links[j]))
        iseven(j) && (ϕ[nthread][j] = δ(links[j-1], links[j], sites[j]))
      end
      ϕ[nthread][N] = ITensor(H_XZ[(pauli[qubit_at(N,N,d)].+1)...,:,:], 
                              sites[N], links[N-1])
      
      Ψ[nthread] = copy(ϕ[nthread])
      
      # contract the bulk
      for i in reverse(2:N-1)
        
        # build intermediate MPO
        links = linkinds(Λ[nthread])
        if iseven(i)
          Λ[nthread][1] = δ(links[1],sites[1],sites[1]')
          for j in 2:N-1
            if iseven(j)
              locE = pauli[qubit_at(i,j,d)] .+ 1
              Λ[nthread][j] = ITensor(V[(pauli[qubit_at(i,j,d)].+1)...,:,:,:,:], 
                                      links[j-1],links[j],sites[j],sites[j]')
            else
              Λ[nthread][j] = δ(links[j-1], links[j], sites[j], sites[j]')
            end
          end
          Λ[nthread][N] = δ(links[N-1],sites[N],sites[N]')
        else
          locE = pauli[qubit_at(i,1,d)] .+ 1
          Λ[nthread][1] = ITensor(H_XXZ[(pauli[qubit_at(i,1,d)].+1)...,:,:,:], 
                                  sites[1], sites[1]', links[1])
          for j in 2:N-1
            if isodd(j)
              locE = pauli[qubit_at(i,j,d)] .+ 1
              Λ[nthread][j] = ITensor(H_XXZZ[(pauli[qubit_at(i,j,d)].+1)...,:,:,:,:],
                                      sites[j], sites[j]', links[j-1], links[j])
            else
              Λ[nthread][j] = δ(links[j-1], links[j], sites[j], sites[j]')
            end
          end
          locE = pauli[qubit_at(i,N,d)] .+ 1
          Λ[nthread][N] = ITensor(H_XXZ[(pauli[qubit_at(i,N,d)].+1)...,:,:,:], 
                                  sites[N], sites[N]', links[N-1])
        end
        
        # contract with previous MPS state
        Ψ[nthread] = noprime(*(Λ[nthread], Ψ[nthread]; method = contract_method, maxdim = maxdim))
      end
      
      # evaluate the coset probability
      for cosetX in [0,1]
        # combine pure error with logical operator
        pauliL = pauli ⊙ logicaloperator([cosetX,0], code)
        
        # build boundary MPS encoding logical (or lack thereof) operation
        links = linkinds(ϕ[nthread]) 
        locE = pauliL[qubit_at(1,1,d)] .+ 1
        ϕ[nthread][1] = ITensor(H_XZ[locE...,:,:], sites[1], links[1]) 
        for j in 2:N-1
          if isodd(j)
            locE = pauliL[qubit_at(1,j,d)] .+ 1
            ϕ[nthread][j] = ITensor(H_XZZ[locE...,:,:,:], sites[j], links[j-1],links[j])
          else
            ϕ[nthread][j] = δ(links[j-1], links[j], sites[j])
          end
        end
        locE = pauliL[qubit_at(1,N,d)] .+ 1
        ϕ[nthread][N] = ITensor(H_XZ[locE...,:,:], sites[N], links[N-1])

        # the coset probability is simply the inner product of the two MPSs
        coset_probability = inner(Ψ[nthread],ϕ[nthread])
        push!(coset_probabilities, coset_probability)
      end
    end
    
    # selecte the coset with maximum likelihood
    ml_coset = argmax(coset_probabilities)
    # generate the most likely logical error
    logical_op = logicaloperator(cosets[ml_coset], code)
    # generate the recovery operation
    recovery = f ⊙ logical_op
    push!(recoveries[nthread], recovery)
  end
  return vcat(recoveries...) 
end
"""
    failure_rate(R::Vector, E::Vector, code::SurfaceCode)

Calculate the logical failure rate of a set of recovery operation
given a set of errors.
"""
function failure_rate(R::Vector, E::Vector, code::SurfaceCode)
  logical_failure_rate = 0.0
  ndata = length(E)
  @assert length(E) == length(R)
  # loop over errors
  for i in 1:length(E)
    # perform recovery
    net_pauli = E[i] ⊙ R[i] 
    # compute wilson loops
    wX,wZ = Wilsonloops(net_pauli, code)
    # check logical error
    (wX == 1 || wZ == 1) && (logical_failure_rate += 1/ndata) 
  end
  return logical_failure_rate
end



"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                                  UTILITIES                                   -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

"""
Addition mod 2
"""
⊙(A::Int, B::Int) = A ⊻ B

⊙(A::Vector{Int64}, B::Vector{Int64}) = A .⊻ B
⊙(A::Tuple{Int64,Int64}, B::Tuple{Int64,Int64}) = A .⊻ B

function ⊙(A::Vector{<:Vector{Int64}},B::Vector{<:Vector{Int64}}) 
  length(A) ≠ length(B) && error("pauli have different lengths")
  return [A[j] ⊙ B[j] for j in 1:length(A)]
end
function ⊙(A::Vector{Tuple{Int64,Int64}},B::Vector{Tuple{Int64,Int64}}) 
  length(A) ≠ length(B) && error("pauli have different lengths")
  return [A[j] ⊙ B[j] for j in 1:length(A)]
end

"""
Generate a pauli chain from a qubit support:
[1,3,5] -> [1,0,1,0,1,...]
"""
support_to_pauli(nqubits::Int64, support::Vector{Int64}) = 
  [n in support ? 1 : 0 for n in 1:nqubits]

support_to_pauli(code::SurfaceCode, support::Vector{Int64}; kwargs...) = 
  support_to_pauli(nqubits(code), support; kwargs...)



#support_to_pauli(nqubits::Int64, support::Tuple) =
#  support_to_pauli(nqubits, [support...])
#
#support_to_topauli(code::SurfaceCode, support::Tuple; kwargs...) = 
#  support_to_pauli(code, [support...])
#"""
#Generate a pauli XZ from X and Z supports
#"""
#function combine_pauliXZ(pauliX::Vector{Int64}, pauliZ::Vector{Int64}) 
#  length(pauliX) ≠ length(pauliZ) && error("pauli have different lengths")
#  return [[pauliX[j],pauliZ[j]] for j in 1:length(pauliX)]
#end
#
#combine_pauliXZ(code::SurfaceCode; kwargs...) = 
#  combineXZ(nqubits(code); kwargs...)


