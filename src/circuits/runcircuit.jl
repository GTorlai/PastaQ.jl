"""
    buildcircuit(M::Union{MPS,MPO}, gates::Vector{<:Tuple};
                 noise = nothing)

Generates a vector of (gate) `ITensor`, from a vector of `Tuple` 
associated with a list of quantum gates. 
If noise is nontrivial, the corresponding Kraus operators are 
added to each gate as a tensor with an extra (Kraus) index.
"""
function buildcircuit(
  M::Union{MPS,MPO},
  circuit::Union{Tuple,Vector{<:Any}};
  noise::Union{Nothing, Tuple, NamedTuple} = nothing
)
  circuit_tensors = ITensor[]
  if circuit isa Tuple
    circuit = [circuit]
  end
  if !isnothing(noise)
    circuit = insertnoise!(circuit, noise)
  end
  for g in circuit
    push!(circuit_tensors, gate(M, g))
  end
  return circuit_tensors
end

function buildcircuit(M::Union{MPS,MPO}, circuit::Vector{Vector{<:Any}}; kwargs...)
  return buildcircuit(M, vcat(circuit...); kwargs...)
end


"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                            TENSOR NETWORK SIMULATOR                                   
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

"""
    runcircuit(M::Union{MPS,MPO}, gate_tensors::Vector{<:ITensor};
               kwargs...)

Apply the circuit to a state (wavefunction/densitymatrix) from a list of tensors.
"""
function runcircuit(
  M::Union{MPS,MPO},
  circuit_tensors::Vector{<:ITensor};
  apply_dag=nothing,
  cutoff=1e-15,
  maxdim=10_000,
  svd_alg="divide_and_conquer",
  move_sites_back::Bool=true,
  kwargs...,
)

  # Check if gate_tensors contains Kraus operators
  inds_sizes = [length(inds(g)) for g in circuit_tensors]
  noiseflag = any(x -> x % 2 == 1, inds_sizes)

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
      return apply(
        circuit_tensors,
        ρ;
        apply_dag=true,
        cutoff=cutoff,
        maxdim=maxdim,
        svd_alg=svd_alg,
        move_sites_back=move_sites_back,
      )

      # Pure state evolution
    else
      # |ψ⟩ -> U |ψ⟩ (MPS -> MPS)
      #  ρ  -> U ρ U† (MPO -> MPO, conjugate evolution)
      if M isa MPS
        return apply(
          circuit_tensors,
          M;
          cutoff=cutoff,
          maxdim=maxdim,
          svd_alg=svd_alg,
          move_sites_back=move_sites_back,
        )
      else
        return apply(
          circuit_tensors,
          M;
          apply_dag=true,
          cutoff=cutoff,
          maxdim=maxdim,
          svd_alg=svd_alg,
          move_sites_back=move_sites_back,
        )
      end
    end
    # Custom mode (apply_dag = true / false)
  else
    if M isa MPO
      # apply_dag = true:  ρ -> U ρ U† (MPO -> MPO, conjugate evolution)
      # apply_dag = false: ρ -> U ρ (MPO -> MPO)
      return apply(
        circuit_tensors,
        M;
        apply_dag=apply_dag,
        cutoff=cutoff,
        maxdim=maxdim,
        svd_alg=svd_alg,
        move_sites_back=move_sites_back,
      )
    elseif M isa MPS
      # apply_dag = true:  ψ -> U ψ -> ρ = (U ψ) (ψ† U†) (MPS -> MPO, conjugate)
      # apply_dag = false: ψ -> U ψ (MPS -> MPS)
      Mc = apply(
        circuit_tensors,
        M;
        cutoff=cutoff,
        maxdim=maxdim,
        svd_alg=svd_alg,
        move_sites_back=move_sites_back,
      )
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
runcircuit(M::Union{MPS, MPO}, circuit::Union{Tuple,Vector{<:Any}};
           noise = nothing, kwargs...) = 
  runcircuit(M, buildcircuit(M, circuit; noise = noise); kwargs...)

function runcircuit(
  M::Union{MPS,MPO},
  circuit::Vector{<:Vector{<:Any}};
  (observer!)=nothing,
  move_sites_back_before_measurements::Bool=false,
  noise=nothing,
  kwargs...,
)

  # is the observer is not provided, apply the vectorized circuit
  isnothing(observer!) && return runcircuit(M, vcat(circuit...); noise=noise, kwargs...)

  # issue warning if there are custom functions and the sites are not being moved back
  if _has_customfunctions(observer!) && move_sites_back_before_measurements == false
    println("--------------")
    println(" WARNING")
    println(
      "\nA custom function is being measured during the gate evolution,\nbut the MPS sites at depth D are not being restored to their \nlocation at depth D-1. If any custom measurement is dependent \non the specific site ordering of the MPS, it should take the \nre-ordering into account. The MPS can be restored to the \ncanonical index ordering by setting:\n\n `move_sites_back_before_measurements = true`\n",
    )
    println("--------------\n")
  end
  M0 = copy(M)
  # record the initial configuration of the indices
  s = siteinds(M)

  for l in 1:length(circuit)
    layer = circuit[l]
    M = runcircuit(M, layer; move_sites_back=move_sites_back_before_measurements, kwargs...)
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
function runcircuit(sites::Vector{<:Index}, 
                circuit::Union{Tuple,Vector{<:Any},Vector{Vector{<:Any}}};
                process::Bool = false, noise = nothing,
                full_representation::Bool = false,
                kwargs...)
  
  
  # Unitary operator for the circuit
  if process && isnothing(noise) 
    U₀ = full_representation ? prod(productoperator(sites)) : productoperator(sites)
    return runcircuit(U₀, circuit; 
                      noise = nothing, 
                      apply_dag = false, 
                      kwargs...) 
  end 
  # Choi matrix
  if process && !isnothing(noise)
    full_representation && error("Exact Choi matrix not yet implemented")
    return choimatrix(sites, circuit; 
                      noise = noise, 
                      kwargs...)
  end
  
  M₀ = full_representation ? prod(productstate(sites)) : productstate(sites) 
  return runcircuit(M₀, circuit; noise = noise, kwargs...) 
end

runcircuit(N::Int, circuit::Any; kwargs...) = 
  runcircuit(siteinds("Qubit", N), circuit; kwargs...)

runcircuit(circuit::Any; kwargs...) = 
  runcircuit(nqubits(circuit), circuit; kwargs...)


"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                               EXACT SIMULATOR                                   
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

"""
    runcircuit(M::ITensor,gate_tensors::Vector{ <: ITensor}; kwargs...)

Apply the circuit to a ITensor from a list of tensors.
"""

function runcircuit(T::ITensor, circuit_tensors::Vector{ <: ITensor}; apply_dag = nothing, kwargs...)
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
      T = is_operator(T) ? T : prime(T, "Site") * dag(T)
      # ρ -> ε(ρ) (MPO -> MPO, conjugate evolution)
      return apply(circuit_tensors, T; apply_dag = true)
    # Pure state evolution
    else
      # |ψ⟩ -> U |ψ⟩ (MPS -> MPS)
      #  ρ  -> U ρ U† (MPO -> MPO, conjugate evolution)
      if !is_operator(T)
        return apply(circuit_tensors, T)
      else
        return apply(circuit_tensors, T; apply_dag = true)
      end
    end
  # Custom mode (apply_dag = true / false)
  else
    Tc = apply(circuit_tensors, T; apply_dag = apply_dag)
    !is_operator(T) && apply_dag && return prime(Tc, "Site") * dag(Tc)
    return Tc
  end
end

"""
    runcircuit(M::ITensor, gates::Vector{<:Tuple})

Apply the circuit to an ITensor from a list of gates.
"""

runcircuit(T::ITensor, circuit::Union{Tuple,Vector{<:Any}}; noise = nothing, kwargs...) =
  runcircuit(T, buildcircuit(T, circuit; noise = noise); kwargs...)

runcircuit(T::ITensor, circuit::Vector{<:Vector{<:Any}}; kwargs...) = 
  runcircuit(T, circuit...; kwargs...)


"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                  CHOI MATRIX                                   
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

"""

Compute the Choi matrix `Λ  = ε ⊗ 1̂(|ξ⟩⟨ξ|)`, where `|ξ⟩= ⨂ⱼ |00⟩ⱼ+|11⟩ⱼ`, 
and `ε`` is a quantum channel built out of a set of quantum gates and 
a local noise model. Returns a MPO with `N` tensor having 4 sites indices. 
"""
choimatrix(circuit::Vector{<:Any}; kwargs...) = 
  choimatrix(nqubits(circuit), gates; kwargs...)

choimatrix(sites::Union{Int, Vector{<:Index}}, args...; kwargs...) = 
  choimatrix(unitary_mpo_to_choi_mpo(productoperator(sites)), args...; kwargs...) 

function choimatrix(Λ0::MPO, circuit::Vector{<:Any};
                    noise = nothing, cutoff = 1e-15, maxdim = 10000,
                    svd_alg = "divide_and_conquer")
  N = length(Λ0)
  if isnothing(noise)
    error("choi matrix requires noise")
  end
  # if circuit has layer structure, transform to vectorized circuit
  circuit = (circuit isa Vector{<:Vector} ? vcat(circuit...) : circuit)
  
  # TODO: simplify by building the circuit directly from the Choi MPO.
  s = [firstind(Λ0[j], tags = "Output") for j in 1:length(Λ0)]
  compiler = productoperator(s)
  prime!(compiler,-1,tags = "Qubit") 
  circuit_tensors = buildcircuit(compiler, circuit; noise = noise)

  # contract to compute the Choi matrix
  return runcircuit(Λ0, circuit_tensors; apply_dag = true, cutoff = cutoff,
                    maxdim = maxdim, svd_alg = svd_alg)
end

