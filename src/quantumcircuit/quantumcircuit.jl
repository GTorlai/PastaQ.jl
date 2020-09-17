"""
  wavefunction(sites::Vector{<:Index})
  wavefunction(N::Int)

Initialize MPS wavefunction `|ψ⟩`
"""

wavefunction(sites::Vector{<:Index}) = productMPS(sites, "0")

wavefunction(N::Int) = wavefunction(siteinds("qubit", N))


""" 
  densitymatrix(sites::Vector{<:Index})
  densitymatrix(N::Int)

Initialize MPO density matrix `ρ`
"""

densitymatrix(sites::Vector{<:Index}) = 
  MPO(productMPS(sites, "0"))

densitymatrix(N::Int) = 
  MPO(siteinds("qubit",N))


"""
  qubits(sites::Vector{<:Index}; mixed::Bool=false)
  qubits(N::Int; mixed::Bool=false)

Initialize qubits to:
- An MPS wavefunction `|ψ⟩` if `mixed=false`
- An MPO density matrix `ρ` if `mixed=true`
"""

qubits(sites::Vector{<:Index}; mixed::Bool=false) = 
  mixed ? densitymatrix(sites) : wavefunction(sites) 

qubits(N::Int; mixed::Bool=false) = qubits(siteinds("qubit", N); mixed=mixed)


""" 
  resetqubits!(M::Union{MPS,MPO})

Reset qubits to the initial state:
- `|ψ⟩=|0,0,…,0⟩` if `M = MPS`
- `ρ = |0,0,…,0⟩⟨0,0,…,0|` if `M = MPO`
"""

function resetqubits!(M::Union{MPS,MPO})
  indices = [firstind(M[j],tags="Site",plev=0) for j in 1:length(M)]
  M_new = (typeof(M) == MPS ? wavefunction(indices) : densitymatrix(indices))
  M[:] = M_new
  return M
end

"""
Initialize a circuit MPO
"""
circuit(sites::Vector{<:Index}) = MPO(sites, "Id")

circuit(N::Int) = circuit(siteinds("qubit", N))

"""----------------------------------------------
                  CIRCUIT FUNCTIONS 
------------------------------------------------- """

"""
  compilecircuit(M::Union{MPS,MPO},gates::Vector{<:Tuple};
                 noise=nothing, kwargs...)

Generates a vector of ITensors from a tuple of gates. 
If noise is nontrivial, the corresponding Kraus operators are 
added to each gate as a tensor with an extra (Kraus) index.
"""
function compilecircuit(M::Union{MPS,MPO},gates::Vector{<:Tuple}; 
                        noise=nothing, kwargs...)
  gate_tensors = ITensor[]
  for g in gates
    push!(gate_tensors, gate(M, g))
    ns = g[2]
    if !isnothing(noise)
      if ns isa Int
        push!(gate_tensors, gate(M, noise, g[2]; kwargs...))
      else
        for n in ns
          push!(gate_tensors, gate(M, noise, n; kwargs...))
        end
      end
    end
  end
  return gate_tensors
end

"""
Apply the circuit to a ITensor from a list of tensors 
"""
runcircuit(M::ITensor,
           gate_tensors::Vector{ <: ITensor};
           kwargs...) =
  apply(gate_tensors, M; kwargs...)

"""
Apply the circuit to a ITensor from a list of gates 
"""
function runcircuit(M::ITensor,gates::Vector{<:Tuple}; cutoff=1e-15,maxdim=10000,kwargs...)
  gate_tensors = compilecircuit(M,gates)
  return runcircuit(M,gate_tensors;cutoff=1e-15,maxdim=10000,kwargs...)
end

"""
Apply the circuit to a state (wavefunction/densitymatrix) from a list of tensors
"""
function runcircuit(M::Union{MPS,MPO},gate_tensors::Vector{<:ITensor}; kwargs...) 
  # Check if gate_tensors contains Kraus operators
  inds_sizes = [length(inds(g)) for g in gate_tensors]
  noiseflag = any(x -> x%2==1 , inds_sizes)
  
  apply_dag = get(kwargs,:apply_dag,nothing)
  cutoff    = get(kwargs,:cutoff,1E-15)
  maxdim    = get(kwargs,:maxdim,10000)

  if apply_dag==false & noiseflag==true
    error("noise simulation requires apply_dag=true")
  end
 
  # Default mode (apply_dag = nothing)
  if isnothing(apply_dag)
    # Noisy evolution: MPS/MPO -> MPO
    if noiseflag
      # If M is an MPS, |ψ⟩ -> ρ = |ψ⟩⟨ψ| (MPS -> MPO)
      ρ = (typeof(M) == MPS ? MPO(M) : M)
      # ρ -> ε(ρ) (MPO -> MPO, conjugate evolution)
      Mc = apply(gate_tensors,ρ; apply_dag=true, cutoff=cutoff,maxdim=maxdim)
    # Pure state evolution
    else
      # |ψ⟩ -> U |ψ⟩ (MPS -> MPS)
      #  ρ  -> U ρ U† (MPO -> MPO, conjugate evolution)
      Mc = (typeof(M) == MPS ? apply(gate_tensors, M; apply_dag=false,cutoff=cutoff,maxdim=maxdim) :
                               apply(gate_tensors, M; apply_dag=true,cutoff=cutoff,maxdim=maxdim))
    end
  # Custom mode (apply_dag = true / false)
  else
    if typeof(M) == MPO
      # apply_dag = true:  ρ -> U ρ U† (MPO -> MPO, conjugate evolution)
      # apply_dag = false: ρ -> U ρ (MPO -> MPO)
      Mc = apply(gate_tensors, M; apply_dag=apply_dag,kwargs...)
    elseif typeof(M) == MPS
      # apply_dag = true:  ψ -> U ψ -> ρ = (U ψ) (ψ† U†) (MPS -> MPO, conjugate)
      # apply_dag = false: ψ -> U ψ (MPS -> MPS)
      Mc = (apply_dag ? MPO(apply(gate_tensors, M; apply_dag=false,cutoff=cutoff,maxdim=maxdim)) :
                        apply(gate_tensors, M; apply_dag=apply_dag=false,cutoff=cutoff,maxdim=maxdim))
    else
      error("Input state must be an MPS or an MPO")
    end
  end
  return Mc
end



"""
    runcircuit(M::Union{MPS,MPO},gates::Vector{<:Tuple}; noise=nothing, apply_dag=nothing, 
               cutoff=1e-15, maxdim=10000, kwargs...)

Apply the circuit to a state (wavefunction or density matrix) from a list of gates.

If an MPS `|ψ⟩` is input, there are three possible modes:

1. By default (`noise = nothing` and `apply_dag = nothing`), the evolution `U|ψ⟩` is performed.
2. If `noise` is set to something nontrivial, the mixed evolution `ε(|ψ⟩⟨ψ|)` is performed.
   Example: `noise = "AD"`, `γ=0.1` (amplitude damping channel with decay rate `γ`)
3. If `noise = nothing` and `apply_dag = true`, the evolution `U|ψ⟩⟨ψ|U†` is performed.

If an MPO `ρ` is input, there are three possible modes:

1. By default (`noise = nothing` and `apply_dag = nothing`), the evolution `U ρ U†` is performed.
2. If `noise` is set to something nontrivial, the evolution `ε(ρ)` is performed.
3. If `noise = nothing` and `apply_dag = false`, the evolution `Uρ` is performed.
"""

function runcircuit(M::Union{MPS,MPO},gates::Vector{<:Tuple}; noise=nothing,apply_dag=nothing, 
                    cutoff=1e-15,maxdim=10000,kwargs...)
  gate_tensors = compilecircuit(M,gates; noise=noise, kwargs...) 
  runcircuit(M,gate_tensors; cutoff=cutoff,maxdim=maxdim,apply_dag=apply_dag, kwargs...)
end



"""
    runcircuit(N::Int, gates::Vector{<:Tuple}; process=false, noise=nothing,
               cutoff=1e-15, maxdim=10000, kwargs...)

Apply the circuit to a state (wavefunction or density matrix) from a list of gates, where the state has `N` physical qubits. 
The starting state is generated automatically based on the flags `process`, `noise`, and `apply_dag`.

1. By default (`noise = nothing`, `apply_dag = nothing`, and `process = false`), 
   the evolution `U|ψ⟩` is performed where the starting state is set to `|ψ⟩ = |000...⟩`. 
   The MPS `U|000...⟩` is returned.
2. If `noise` is set to something nontrivial, the mixed evolution `ε(|ψ⟩⟨ψ|)` is performed, 
   where the starting state is set to `|ψ⟩ = |000...⟩`. 
   The MPO `ε(|000...⟩⟨000...|)` is returned.
3. If `process = true`, the evolution `U 1̂` is performed, where the starting state `1̂ = (1⊗1⊗1⊗…⊗1)`. 
   The MPO approximation for the unitary represented by the set of gates is returned. 
   In this case, `noise` must be `nothing`.
"""

function runcircuit(N::Int,gates::Vector{<:Tuple}; process=false,noise=nothing,
                    cutoff=1e-15,maxdim=10000,kwargs...)
  if process==false
    ψ = qubits(N) # = |0,0,0,…,0⟩
    # noiseless: ψ -> U ψ
    # noisy:     ψ -> ρ = ε(|ψ⟩⟨ψ|)
    return runcircuit(ψ,gates;noise=noise,cutoff=cutoff,maxdim=maxdim,kwargs...)
  
  elseif process==true & isnothing(noise)
    if isnothing(noise)
      U = circuit(N) # = 1⊗1⊗1⊗…⊗1
      return runcircuit(U,gates;noise=nothing,apply_dag=false,cutoff=cutoff,maxdim=maxdim,kwargs...) 
    else
      error("Cannot build the circuit MPO if noise!=nothing")
    end
  end
    
end

"""
    choimatrix(N::Int, gates::Vector{<:Tuple}; noise=nothing, apply_dag=false,
               cutoff=1e-15, maxdim=10000, kwargs...)

Compute the Choi matrix `Λ` from a set of gates that make up a quantum process.

If `noise = nothing` (the default), for an N-qubit process, by default the square 
root of the Choi matrix `|U⟩` is returned, such that the Choi matrix is the rank-1 matrix 
`Λ = |U⟩⟨U|`. `|U⟩` is an MPS with `2N` sites for a process running on `N` physical qubits. 
It is the "state" version of the unitary approximation for the full gate evolution `U`.

If `noise != nothing`, an approximation for the Choi matrix is returned as an MPO 
with `2N` sites, for a process with `N` physical qubits.

If `noise = nothing` and `apply_dag = true`, the Choi matrix `Λ` is returned as an MPO with 
`2N` sites. In this case, the MPO `Λ` is equal to `|U⟩⟨U|`.

#TODO: Explain site ordering and normalization
"""

function choimatrix(N::Int,gates::Vector{<:Tuple};noise=nothing,apply_dag=false,
                    cutoff=1e-15,maxdim=10000,kwargs...)
  if isnothing(noise)
    # Get circuit MPO
    U = runcircuit(N,gates;process=true,cutoff=1e-15,maxdim=10000,kwargs...)
    
    # Choi indices 
    addtags!(U,"Input", plev=0,tags="qubit")
    addtags!(U,"Output",plev=1,tags="qubit")
    noprime!(U)
    # SVD to bring into 2N-sites MPS
    Λ0 = splitchoi(U,noise=nothing,cutoff=cutoff,maxdim=maxdim)
    # if apply_dag = true:  Λ = |U⟩⟩ ⟨⟨U†|
    # if apply_dag = false: Λ = |U⟩⟩
    Λ = (apply_dag ? MPO(Λ0) : Λ0)
  else
    # Initialize circuit MPO
    U = circuit(N)
    addtags!(U,"Input",plev=0,tags="qubit")
    addtags!(U,"Output",plev=1,tags="qubit")
    prime!(U,tags="Input")
    prime!(U,tags="Link")
    
    s = [siteinds(U,tags="Output")[j][1] for j in 1:length(U)]
    compiler = circuit(s)
    prime!(compiler,-1,tags="qubit") 
    gate_tensors = compilecircuit(compiler, gates; noise=noise, kwargs...)

    M = ITensor[]
    push!(M,U[1] * noprime(U[1]))
    Cdn = combiner(inds(M[1],tags="Link")[1],inds(M[1],tags="Link")[2],
                  tags="Link,n=1")
    M[1] = M[1] * Cdn
    for j in 2:N-1
      push!(M,U[j] * noprime(U[j]))
      Cup = Cdn
      Cdn = combiner(inds(M[j],tags="Link,n=$j")[1],inds(M[j],tags="Link,n=$j")[2],tags="Link,n=$j")
      M[j] = M[j] * Cup * Cdn
    end
    push!(M, U[N] * noprime(U[N]))
    M[N] = M[N] * Cdn
    ρ = MPO(M)
    Λ0 = runcircuit(ρ,gate_tensors;apply_dag=true,cutoff=cutoff, maxdim=maxdim)
    Λ = splitchoi(Λ0,noise=noise,cutoff=cutoff,maxdim=maxdim)
  end
  return Λ
end

function splitchoi(Λ::MPO;noise=nothing,cutoff=1e-15,maxdim=1000)
  T = ITensor[]
  if isnothing(noise)
    u,S,v = svd(Λ[1],firstind(Λ[1],tags="Input"), 
                cutoff=cutoff, maxdim=maxdim)
  else
    u,S,v = svd(Λ[1],inds(Λ[1],tags="Input"), 
                cutoff=cutoff, maxdim=maxdim)
  end
  push!(T,u*S)
  push!(T,v)
  
  for j in 2:length(Λ)
    if isnothing(noise)
      u,S,v = svd(Λ[j],firstind(Λ[j],tags="Input"),commonind(Λ[j-1],Λ[j]),
                  cutoff=cutoff,maxdim=maxdim)
    else
      u,S,v = svd(Λ[j],inds(Λ[j],tags="Input")[1],inds(Λ[j],tags="Input")[2],
                  commonind(Λ[j-1],Λ[j]),cutoff=cutoff,maxdim=maxdim) 
    end
    push!(T,u*S)
    push!(T,v)
  end
  Λ_split = (isnothing(noise) ? MPS(T) : MPO(T))
  return Λ_split
end

