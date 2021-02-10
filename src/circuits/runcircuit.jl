"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                                QUANTUM STATES                                -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

"""
    trivialstate(N::Int; mixed::Bool=false)
    
    trivialstate(sites::Vector{<:Index}; mixed::Bool=false)


Initialize qubits to:
- An MPS wavefunction `|ψ⟩` if `mixed = false`
- An MPO density matrix `ρ` if `mixed = true`
"""
trivialstate(N::Int; mixed::Bool = false) =
  trivialstate(siteinds("Qubit", N); mixed = mixed)

function trivialstate(sites::Vector{<:Index}; mixed::Bool = false)
  ψ = productMPS(sites, "0")
  mixed && return MPO(ψ)
  return ψ
end

"""
    trivialstate(M::Union{MPS,MPO,LPDO}; mixed::Bool=false)

Initialize qubits on the Hilbert space of a reference state,
given as `MPS`, `MPO` or `LPDO`.
"""
trivialstate(M::Union{MPS,MPO,LPDO}; mixed::Bool = false) =
  trivialstate(hilbertspace(M); mixed = mixed)


"""
    trivialstate(N::Int, states::Vector{String}; mixed::Bool=false)

    trivialstate(sites::Vector{<:Index}, states::Vector{String};mixed::Bool = false)
    
    trivialstate(M::Union{MPS,MPO,LPDO}, states::Vector{String}; mixed::Bool = false) 

Initialize the qubits to a given single-qubit product state.
"""
trivialstate(N::Int, states::Vector{String}; mixed::Bool = false) =
  trivialstate(siteinds("Qubit", N), states; mixed = mixed)

trivialstate(M::Union{MPS,MPO,LPDO}, states::Vector{String}; mixed::Bool = false) = 
  trivialstate(hilbertspace(M), states; mixed = mixed)

function trivialstate(sites::Vector{<:Index}, states::Vector{String};
                      mixed::Bool = false)
  N = length(sites)
  @assert N == length(states)

  ψ = productMPS(sites, "0")

  if N == 1
    s1 = sites[1]
    state1 = state(states[1])
    if eltype(state1) <: Complex
      ψ[1] = complex(ψ[1])
    end
    for j in 1:dim(s1)
      ψ[1][s1 => j] = state1[j]
    end
    mixed && return MPO(ψ)
    return ψ
  end

  # Set first site
  s1 = sites[1]
  l1 = linkind(ψ, 1)
  state1 = state(states[1])
  if eltype(state1) <: Complex
    ψ[1] = complex(ψ[1])
  end
  for j in 1:dim(s1)
    ψ[1][s1 => j, l1 => 1] = state1[j]
  end

  # Set sites 2:N-1
  for n in 2:N-1
    sn = sites[n]
    ln_1 = linkind(ψ, n-1)
    ln = linkind(ψ, n)
    state_n = state(states[n])
    if eltype(state_n) <: Complex
      ψ[n] = complex(ψ[n])
    end
    for j in 1:dim(sn)
      ψ[n][sn => j, ln_1 => 1, ln => 1] = state_n[j]
    end
  end
  
  # Set last site N
  sN = sites[N]
  lN_1 = linkind(ψ, N-1)
  state_N = state(states[N])
  if eltype(state_N) <: Complex
    ψ[N] = complex(ψ[N])
  end
  for j in 1:dim(sN)
    ψ[N][sN => j, lN_1 => 1] = state_N[j]
  end
  
  mixed && return MPO(ψ)
  return ψ
end

"""
    trivialprocess(N::Int; mixed::Bool = false)

    trivialprocess(sites::Vector{<:Index}; mixed::Bool=false)

Initialize a circuit MPO
"""
trivialprocess(N::Int;  mixed::Bool = false) = 
  trivialprocess(siteinds("Qubit", N); mixed = mixed)

trivialprocess(M::Union{MPS, MPO,LPDO}; mixed::Bool = false) = 
  trivialprocess(hilbertspace(M); mixed = mixed)

function trivialprocess(sites::Vector{<:Index}; mixed::Bool = false) 
  U = MPO(sites, "Id")
  !mixed && return U
  
  N = length(U)
  U = prime(choitags(U))
  
  M = ITensor[]
  push!(M,U[1] * noprime(U[1]))
  if N > 1
    Cdn = combiner(inds(M[1], tags = "Link")[1],inds(M[1], tags = "Link")[2],
                  tags = "Link,l=1")
    M[1] = M[1] * Cdn
    for j in 2:N-1
      push!(M,U[j] * noprime(U[j]))
      Cup = Cdn
      Cdn = combiner(inds(M[j], tags = "Link,l=$j")[1], 
                     inds(M[j], tags = "Link,l=$j")[2], 
                     tags="Link,l=$j")
      M[j] = M[j] * Cup * Cdn
    end
    push!(M, U[N] * noprime(U[N]))
    M[N] = M[N] * Cdn
  end
  return MPO(M)
end


"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                              QUANTUM CIRCUITS                                -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""



"""
    buildcircuit(M::Union{MPS,MPO}, gates::Vector{<:Tuple};
                 noise = nothing)

Generates a vector of (gate) `ITensor`, from a vector of `Tuple` 
associated with a list of quantum gates. 
If noise is nontrivial, the corresponding Kraus operators are 
added to each gate as a tensor with an extra (Kraus) index.
"""
function buildcircuit(M::Union{MPS,MPO}, circuit::Union{Tuple,Vector{<:Any}}; 
                      noise::Union{Nothing, String, Tuple{String, NamedTuple}} = nothing)
  circuit_tensors = ITensor[]
  if circuit isa Tuple
    circuit = [circuit]
  end
  for g in circuit
    push!(circuit_tensors, gate(M, g))
    ns = g[2]
    if !isnothing(noise)
      for n in ns
        if noise isa String
          noisegate = (noise, n)
        elseif noise isa Tuple{String, NamedTuple}
          noisegate = (noise[1], n, noise[2])
        end
        push!(circuit_tensors, gate(M, noisegate))
      end
    end
  end
  return circuit_tensors
end

buildcircuit(M::Union{MPS,MPO}, circuit::Vector{Vector{<:Any}}; kwargs...) = 
  buildcircuit(M, vcat(circuit...); kwargs...)

"""
    runcircuit(M::Union{MPS,MPO}, gate_tensors::Vector{<:ITensor};
               kwargs...)

Apply the circuit to a state (wavefunction/densitymatrix) from a list of tensors.
"""
function runcircuit(M::Union{MPS, MPO},
                    circuit_tensors::Vector{<:ITensor};
                    apply_dag = nothing,
                    cutoff = 1e-15,
                    maxdim = 10_000,
                    svd_alg = "divide_and_conquer",
                    move_sites_back::Bool = true,
                    kwargs...)

  # Check if gate_tensors contains Kraus operators
  inds_sizes = [length(inds(g)) for g in circuit_tensors]
  noiseflag = any(x -> x % 2 == 1 , inds_sizes)

  if apply_dag == false && noiseflag == true
    error("noise simulation requires apply_dag=true")
  end
 
  # Default mode (apply_dag = nothing)
  if isnothing(apply_dag)
    # Noisy evolution: MPS/MPO -> MPO
    if noiseflag
      # If M is an MPS, |ψ⟩ -> ρ = |ψ⟩⟨ψ| (MPS -> MPO)
      ρ = (typeof(M) == MPS ? MPO(M) : M)
      # ρ -> ε(ρ) (MPO -> MPO, conjugate evolution)
      return apply(circuit_tensors, ρ; apply_dag = true,
                   cutoff = cutoff, maxdim = maxdim,
                   svd_alg = svd_alg,
                   move_sites_back = move_sites_back)
      
    # Pure state evolution
    else
      # |ψ⟩ -> U |ψ⟩ (MPS -> MPS)
      #  ρ  -> U ρ U† (MPO -> MPO, conjugate evolution)
      if M isa MPS
        return apply(circuit_tensors, M; cutoff = cutoff,
                     maxdim = maxdim, svd_alg = svd_alg,
                     move_sites_back = move_sites_back)
      else
        return apply(circuit_tensors, M; apply_dag = true,
                     cutoff = cutoff, maxdim = maxdim,
                     svd_alg = svd_alg,
                     move_sites_back = move_sites_back)
      end
    end
  # Custom mode (apply_dag = true / false)
  else
    if M isa MPO
      # apply_dag = true:  ρ -> U ρ U† (MPO -> MPO, conjugate evolution)
      # apply_dag = false: ρ -> U ρ (MPO -> MPO)
      return apply(circuit_tensors, M; apply_dag = apply_dag,
                   cutoff = cutoff, maxdim = maxdim,
                   svd_alg = svd_alg,
                   move_sites_back = move_sites_back)
    elseif M isa MPS
      # apply_dag = true:  ψ -> U ψ -> ρ = (U ψ) (ψ† U†) (MPS -> MPO, conjugate)
      # apply_dag = false: ψ -> U ψ (MPS -> MPS)
      Mc = apply(circuit_tensors, M; cutoff = cutoff, maxdim = maxdim,
                 svd_alg = svd_alg,
                 move_sites_back = move_sites_back)
      if apply_dag
        Mc = MPO(Mc)
      end
      return Mc
    else
      error("Input state must be an MPS or an MPO")
    end
  end
end


"""

Compute the Choi matrix `Λ  = ε ⊗ 1̂(|ξ⟩⟨ξ|)`, where `|ξ⟩= ⨂ⱼ |00⟩ⱼ+|11⟩ⱼ`, 
and `ε`` is a quantum channel built out of a set of quantum gates and 
a local noise model. Returns a MPO with `N` tensor having 4 sites indices. 
"""
choimatrix(circuit::Vector{<:Any}; kwargs...) = 
  choimatrix(nqubits(circuit), gates; kwargs...)

choimatrix(N::Int, args...; kwargs...) = 
  choimatrix(trivialprocess(N; mixed = true), args...; kwargs...) 

choimatrix(sites::Vector{<:Index}, args...; kwargs...)  =
  choimatrix(trivialprocess(sites; mixed = true), args...; kwargs...)

function choimatrix(Λ0::MPO, circuit::Vector{<:Any};
                    noise = nothing, cutoff = 1e-15, maxdim = 10000,
                    svd_alg = "divide_and_conquer")
  N = length(Λ0)
  if isnothing(noise)
    error("choi matrix requires noise")
  end
  # if circuit has layer structure, transform to vectorized circuit
  circuit = (circuit isa Vector{<:Vector} ? vcat(circuit...) : circuit)
  
  s = [firstind(Λ0[j], tags = "Output") for j in 1:length(Λ0)]
  compiler = trivialprocess(s)
  prime!(compiler,-1,tags = "Qubit") 
  circuit_tensors = buildcircuit(compiler, circuit; noise = noise)
  # contract to compute the Choi matrix
  return runcircuit(Λ0, circuit_tensors; apply_dag = true, cutoff = cutoff,
                    maxdim = maxdim, svd_alg = svd_alg)
end


"""
     runcircuit(M::Union{MPS, MPO}, gates::Union{Tuple,Vector{<:Any}};
                noise = nothing, kwargs...)

    runcircuit(M::Union{MPS, MPO}, gates::Vector{<:Vector{<:Any}};
               observer! = nothing, 
               move_sites_back_before_measurements::Bool = false,
               noise = nothing,
               kwargs...)

Apply a quantum circuit to a state (wavefunction or density matrix) 

If an MPS `|ψ⟩` is input, there are three possible modes:

1. By default (`noise = nothing` and `apply_dag = nothing`), the evolution `U|ψ⟩` is performed.
2. If `noise` is set to something nontrivial, the mixed evolution `ε(|ψ⟩⟨ψ|)` is performed.
   Example: `noise = ("amplitude_damping", (γ = 0.1,))` (amplitude damping channel with decay rate `γ = 0.1`)
3. If `noise = nothing` and `apply_dag = true`, the evolution `U|ψ⟩⟨ψ|U†` is performed.

If an MPO `ρ` is input, there are three possible modes:

1. By default (`noise = nothing` and `apply_dag = nothing`), the evolution `U ρ U†` is performed.
2. If `noise` is set to something nontrivial, the evolution `ε(ρ)` is performed.
3. If `noise = nothing` and `apply_dag = false`, the evolution `Uρ` is performed.

If an `Observer` is provided as input, and `circuit` is made out of a sequence
of layers of gates, performs a measurement of the observables contained in
Observer, after the application of each layer.
"""
function runcircuit(M::Union{MPS, MPO}, circuit::Union{Tuple,Vector{<:Any}};
                    noise = nothing, kwargs...)
  circuit_tensors = buildcircuit(M, circuit; noise = noise) 
  return runcircuit(M, circuit_tensors; kwargs...)
end

function runcircuit(M::Union{MPS, MPO}, circuit::Vector{<:Vector{<:Any}};
                    observer! = nothing, 
                    move_sites_back_before_measurements::Bool = false,
                    noise = nothing,
                    kwargs...)
                  
  # is the observer is not provided, apply the vectorized circuit
  isnothing(observer!) && return runcircuit(M, vcat(circuit...); noise = noise, kwargs...)
  
  # issue warning if there are custom functions and the sites are not being moved back
  if _has_customfunctions(observer!) && move_sites_back_before_measurements == false
    println("--------------")
    println(" WARNING")
    println("\nA custom function is being measured during the gate evolution,\nbut the MPS sites at depth D are not being restored to their \nlocation at depth D-1. If any custom measurement is dependent \non the specific site ordering of the MPS, it should take the \nre-ordering into account. The MPS can be restored to the \ncanonical index ordering by setting:\n\n `move_sites_back_before_measurements = true`\n")
    println("--------------\n")
  end
  M0 = copy(M)
  # record the initial configuration of the indices
  s = siteinds(M)

  for l in 1:length(circuit)
    layer = circuit[l]
    M = runcircuit(M, layer; move_sites_back = move_sites_back_before_measurements, kwargs...)
    if !isnothing(observer!)
      measure!(observer!, M, s)
    end
  end  
  if move_sites_back_before_measurements == false
    new_s = siteinds(M)
    ns = 1:length(M)
    ñs = [findsite(M0, i) for i in new_s]
    M = movesites(M, ns .=> ñs; kwargs...)
  end
  return M
end





"""
    runcircuit(N::Int, gates::Vector{<:Tuple};
               process = false,
               noise = nothing,
               cutoff = 1e-15,
               maxdim = 10000,
               svd_alg = "divide_and_conquer")

Run the circuit corresponding to a list of quantum gates on a system of `N` qubits. 
The starting state is generated automatically based on the flags `process`, `noise`, and `apply_dag`.

1. By default (`noise = nothing`, `apply_dag = nothing`, and `process = false`), 
   the evolution `U|ψ⟩` is performed where the starting state is set to `|ψ⟩ = |000...⟩`. 
   The MPS `U|000...⟩` is returned.
2. If `noise` is set to something nontrivial, the mixed evolution `ε(|ψ⟩⟨ψ|)` is performed, 
   where the starting state is set to `|ψ⟩ = |000...⟩`. 
   The MPO `ε(|000...⟩⟨000...|)` is returned.
3. If `process = true` and `noise = nothing`, the evolution `U 1̂` is performed, 
   where the starting state `1̂ = (1⊗1⊗1⊗…⊗1)`. The MPO approximation for the unitary 
   represented by the set of gates is returned.
4. If `process = true` and `noise` is set to something nontrivial, the function returns the Choi matrix 
   `Λ = ε⊗1̂(|ξ⟩⟨ξ|)`, where `|ξ⟩= ⨂ⱼ |00⟩ⱼ+|11⟩ⱼ`, approximated by a MPO with 4 site indices,
   two for the input and two for the output Hilbert space of the quantum channel.
"""
function runcircuit(sites::Vector{<:Index}, circuit::Union{Tuple,Vector{<:Any},Vector{Vector{<:Any}}};
                    process = false, noise = nothing, kwargs...)
  
  (process && isnothing(noise)) && return runcircuit(trivialprocess(sites), circuit; 
                                                     noise = nothing, 
                                                     apply_dag = false, 
                                                     kwargs...) 
  
  (process && !isnothing(noise)) && return choimatrix(sites, circuit; 
                                                      noise = noise, 
                                                      kwargs...)
  
  return runcircuit(trivialstate(sites), circuit; noise = noise, kwargs...) 
end

runcircuit(N::Int, args...; kwargs...) = 
  runcircuit(siteinds("Qubit", N), args...; kwargs...)

runcircuit(circuit::Any, args...; kwargs...) = 
  runcircuit(nqubits(circuit), circuit,  args...; kwargs...)




"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                                  UTILITIES                                   -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""



"""
    runcircuit(M::ITensor,gate_tensors::Vector{ <: ITensor}; kwargs...)

Apply the circuit to a ITensor from a list of tensors.
"""

runcircuit(M::ITensor, circuit_tensors::Vector{ <: ITensor}; kwargs...) =
  apply(circuit_tensors, M; kwargs...)

"""
    runcircuit(M::ITensor, gates::Vector{<:Tuple})

Apply the circuit to an ITensor from a list of gates.
"""

runcircuit(M::ITensor, circuit::Vector{<:Any }; noise = nothing, kwargs...) =
  runcircuit(M, buildcircuit(M, circuit; noise = noise); kwargs...)

