@doc raw"""
    buildcircuit(
      hilbert::Vector{<:Index},
      circuit::Union{Tuple, Vector{<:Any}};
      noise::Union{Nothing, Tuple, NamedTuple} = nothing
    )
    buildcircuit(M::Union{MPS,MPO,ITensor}, args...; kwargs...)

Compile a circuit from a lazy representation into a vector of `ITensor`.
For example, a gate element of `circuit`, `("gn", (i,j))` is turned into a rrank-4 tensor corresponding to the `i` and `j` element of `hilbert`.

If `noise` is passed, the corresponding Kraus operator are inserted appropriately
after the gates in the circuit.
"""
function buildcircuit(
  hilbert::Vector{<:Index},
  circuit::Union{Tuple,Vector{<:Any}};
  noise::Union{Nothing,Tuple,NamedTuple}=nothing,
  eltype=nothing,
  device=identity,
)
  circuit_tensors = ITensor[]
  circuit = circuit isa Tuple ? [circuit] : circuit
  circuit = insertnoise(circuit, noise)
  circuit_tensors = isempty(circuit) ? ITensor[] : [gate(hilbert, g) for g in circuit]
  circuit_tensors = device(_convert_leaf_eltype(eltype, circuit_tensors))
  return circuit_tensors
end

function buildcircuit(hilbert::Vector{<:Index}, circuit::Vector{<:Vector{<:Any}}; kwargs...)
  circuit_tensors = Vector{Vector{ITensor}}()
  for layer in circuit
    circuit_tensors = vcat(circuit_tensors, [buildcircuit(hilbert, layer; kwargs...)])
  end
  return circuit_tensors
end

function buildcircuit(M::Union{MPS,MPO,ITensor}, args...; kwargs...)
  return buildcircuit(originalsiteinds(M), args...; kwargs...)
end

"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                            TENSOR NETWORK SIMULATOR
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

@doc raw"""
    runcircuit(circuit::Any; kwargs...)

Execute quantum circuit (see below).
"""
runcircuit(circuit::Any; kwargs...) = runcircuit(nqubits(circuit), circuit; kwargs...)

runcircuit(N::Int, circuit::Any; kwargs...) = runcircuit(qubits(N), circuit; kwargs...)

@doc raw"""
    runcircuit(hilbert::Vector{<:Index}, circuit::Tuple; kwargs...)

Execute quantum circuit on Hilbert space `hilbert` (see below).
"""
function runcircuit(hilbert::Vector{<:Index}, circuit::Tuple; kwargs...)
  return runcircuit(hilbert, [circuit]; kwargs...)
end

@doc raw"""
    runcircuit(hilbert::Vector{<:Index}, circuit::Vector;
               full_representation::Bool = false,
               process::Bool = false,
               noise = nothing,
               kwargs...)

    runcircuit(M::Union{MPS, MPO, ITensor}, circuit::Union{Tuple, AbstractVector};
               full_representation::Bool = false, noise = nothing, kwargs...)

Run the circuit corresponding to a list of quantum gates on a system of ``n`` qubits,
with input Hilbert space `hilbert`. The specific method of this general function is
specified by the keyword arguments `process` and `noise`.

By default (`process = false` and `noise = nothing`), `runcircuit` returns an MPS
wavefunction corresponding to the contraction of each quantum gate in `circuit` with
the zero product state

```math
|\psi\rangle = U_M\dots U_2 U_1|0,0,\dots,0\rangle
```

If `process = true`, the output is the MPO corresponding to the full unitary circuit:
```math
U = U_M\dots U_2 U_1
```

If `noise` is set to a given input noise model (in lazy representation), Kraus operators
are added to each gate in the circuit, and the output is the MPO density operator given by
the contraction of the noisy circuit with input zero state:

```math
\rho = \mathcal{E}(|0,\dots,0\rangle\langle0,\dots,0|)
```

Finally, if both `noise = ...` and `proces  = true`, the output is the full quantum channel, which
we by default represent with its Choi matrix:

```math
\Lambda = (1+\mathcal{E})|\Phi\rangle\langle\Phi|^{\otimes n}
```

If `full_representation = true`, the contraction is performed without approximation,
leading to an output object whose size scales exponentiall with ``n``
"""
function runcircuit(
  hilbert::Vector{<:Index},
  circuit::Vector;
  full_representation::Bool=false,
  process::Bool=false,
  noise=nothing,
  eltype=nothing,
  device=identity,
  kwargs...,
)

  # this step is required to check whether there is already noise in the circuit
  # which was added using the `insertnoise` function. If so, one should call directly
  # the `choimatrix` function.
  circuit_tensors = buildcircuit(hilbert, circuit; noise, device, eltype)
  layers = circuit_tensors isa Vector{<:ITensor} ? (circuit_tensors,) : circuit_tensors
  noiseflag = any(isodd, (length(inds(g)) for layer in layers for g in layer))

  # Unitary operator for the circuit
  if process && !noiseflag
    U₀ = productoperator(hilbert; eltype, device)
    U₀ = full_representation ? convert_to_full_representation(U₀) : U₀
    return runcircuit(U₀, circuit_tensors; apply_dag=false, kwargs...)
  end
  # Choi matrix
  if process && noiseflag
    return choimatrix(
      hilbert, vcat(circuit_tensors...); full_representation, eltype, device, kwargs...
    )
  end

  M₀ = productstate(hilbert; eltype, device)
  M₀ = full_representation ? convert_to_full_representation(M₀) : M₀
  return runcircuit(M₀, circuit_tensors; kwargs...)
end

function runcircuit(
  M::Union{MPS,MPO,ITensor},
  circuit::Union{Tuple,AbstractVector};
  full_representation::Bool=false,
  noise=nothing,
  eltype=nothing,
  device=identity,
  kwargs...,
)
  M = full_representation ? convert_to_full_representation(M) : M
  return runcircuit(M, buildcircuit(M, circuit; noise, eltype, device); kwargs...)
end

@doc raw"""
    runcircuit(
      M::Union{MPS, MPO, ITensor},
      circuit::Vector{<:Vector{<:ITensor}};
      (observer!)=nothing,
      move_sites_back_before_measurements::Bool=false,
      noise = nothing,
      outputlevel = 1,
      outputpath = nothing,
      savestate = false,
      print_metrics = [],
      kwargs...)

Apply a quantum circuit to an input state `M`, where the circuit is built out of a
sequence of layers of quantum gates. The input state may be an MPS wavefunction
``|\psi\rangle``, an MPO density operator ``ρ`` (or unitary operator ``U``), etc.

By feeding a "layered" circuit, we can enable measurement and keep track of metrics
as a function of the circuit's depth.

Other than the keyword arguments of the high-level interface, here we can provide:
+ `(observer!)`: observer object (from Observers.jl).
+ `outputlevel = 1`: amount of printing during calculation.
+ `outputpath = nothing`: if set, save observer on file.
+ `savestate = false`: if `true`, save the `MPS/MPO` on file.
+ `print_metrics = []`: the metrics in the `observed` to print at each depth.`
"""
function runcircuit(
  M::Union{MPS,MPO,ITensor},
  circuit::Vector{<:Vector{<:ITensor}};
  (observer!)=nothing,
  move_sites_back_before_measurements::Bool=false,
  noise=nothing,
  eltype=nothing,
  device=identity,
  outputlevel=1,
  outputpath=nothing,
  savestate=false,
  print_metrics=[],
  kwargs...,
)
  M = device(_convert_leaf_eltype(eltype, M))
  circuit = device(_convert_leaf_eltype(eltype, circuit))
  # is the observer is not provided, apply the vectorized circuit
  isnothing(observer!) && return runcircuit(M, vcat(circuit...); noise, kwargs...)

  # issue warning if there are custom functions and the sites are not being moved back
  if !isnothing(observer!) && !move_sites_back_before_measurements
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
  if !isnothing(observer!)
    update!(observer!, M; sites=s)
  end
  for l in 1:length(circuit)
    layer = circuit[l]
    t = @elapsed begin
      M = runcircuit(
        M,
        layer; #noise = noise,
        move_sites_back=move_sites_back_before_measurements,
        kwargs...,
      )
      if !isnothing(observer!)
        update!(observer!, M; sites=s)
      end
    end
    if outputlevel ≥ 1
      @printf("%-4d  ", l)
      @printf("χ = %-4d  ", maxlinkdim(M))
      #TODO add the truncation error here
      printobserver(observer!, print_metrics)
      @printf("elapsed = %-4.3fs", t)
      println()
    end
    if !isnothing(outputpath)
      observerpath = outputpath * "_observer.jld2"
      save(observerpath; observer!)
      if savestate
        statepath = outputpath * "_state.h5"
        h5rewrite(statepath) do fout
          write(fout, "state", M)
        end
      end
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

@doc raw"""
    runcircuit(
      M::Union{MPS,MPO},
      circuit_tensors::Vector{<:ITensor};
      apply_dag=nothing,
      cutoff=1e-15,
      maxdim=10_000,
      svd_alg="divide_and_conquer",
      move_sites_back::Bool=true,
      kwargs...)

Apply a set of "gate" tensors (alredy in the form of `ITensor`) to an input
state `M`, with options:
+ `apply_dag = nothing`: whether to perform conjugate evolution.
+ `cutoff = 1e-15`: truncation cutoff in SVD.
+ `maxdim = 10_000`: maximum bond dimension at SVD trunctions.
+ `svd_alg = "divide_and_conquer"`: SVD algorithm (see ITensors.jl).
+ `move_sites_back = true`: move sites back after long-range gate.

By default, `apply_dag = nothing` and the interface is dictated by the input state,
and whether or not the vector of `ITensor` containins rank-3 noisy tensors (i.e. Kraus operators).

For an input MPS ``|\psi_0\rangle``, with a unitary circuit, the output is
```math
|\psi\rangle = U_M\dots U_2 U_1|\psi_0\rangle
```
while for noisy circuits:
```math
\rho = \mathcal{E}(|\psi_0\rangle\langle\psi_0|)
```

For an input MPO ``\rho_0``, the output is
```math
\rho = U_M\dots U_2 U_1 \rho_0 U^\dagger_1 U^\dagger_2,\dots,U^\dagger_M
```
for unitary circuits, and ``\rho = \mathcal{E}(\rho_0)`` for noisy circuits.
"""
function runcircuit(
  M::Union{MPS,MPO},
  circuit_tensors::Vector{<:ITensor};
  apply_dag=nothing,
  cutoff=1e-15,
  maxdim=10_000,
  svd_alg="divide_and_conquer",
  move_sites_back::Bool=true,
  eltype=nothing,
  device=identity,
  kwargs...,
)
  M = device(_convert_leaf_eltype(eltype, M))
  circuit_tensors = device(_convert_leaf_eltype(eltype, circuit_tensors))

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
      #XXX to be differentiated
      if typeof(M) == MPS
        ρ = outer(M', M)
      else
        ρ = M
      end
      # ρ -> ε(ρ) (MPO -> MPO, conjugate evolution)
      return apply(
        circuit_tensors, ρ; apply_dag=true, cutoff, maxdim, svd_alg, move_sites_back
      )
      # Pure state evolution
    else
      # |ψ⟩ -> U |ψ⟩ (MPS -> MPS)
      #  ρ  -> U ρ U† (MPO -> MPO, conjugate evolution)
      if M isa MPS
        return apply(circuit_tensors, M; cutoff, maxdim, svd_alg, move_sites_back)
      else
        return apply(
          circuit_tensors, M; apply_dag=true, cutoff, maxdim, svd_alg, move_sites_back
        )
      end
    end
    # Custom mode (apply_dag = true / false)
  else
    if M isa MPO
      # apply_dag = true:  ρ -> U ρ U† (MPO -> MPO, conjugate evolution)
      # apply_dag = false: ρ -> U ρ (MPO -> MPO)
      return apply(circuit_tensors, M; apply_dag, cutoff, maxdim, svd_alg, move_sites_back)
    elseif M isa MPS
      # apply_dag = true:  ψ -> U ψ -> ρ = (U ψ) (ψ† U†) (MPS -> MPO, conjugate)
      # apply_dag = false: ψ -> U ψ (MPS -> MPS)
      Mc = apply(circuit_tensors, M; cutoff, maxdim, svd_alg, move_sites_back)
      if apply_dag
        Mc = outer(Mc', Mc)
      end
      return Mc
    else
      error("Input state must be an MPS or an MPO")
    end
  end
end

"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                                  CHOI MATRIX
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

@doc raw"""
    choimatrix(circuit::Vector{<:Any}; kwargs...) =
    choimatrix(sites::Vector{<:Index}, circuit::Vector{<:Any}; kwargs...)
    choimatrix(sites::Vector{<:Index}, circuit_tensors::Vector{<:ITensor};
               full_representation = false,kwargs...)

Compute the Choi matrix for a noisy channel

```math
\Lambda = (1+\mathcal{E})|\Phi\rangle\langle\Phi|^{\otimes n}
```

If `full_representation = true`, the contraction is performed without approximation,
leading to an output object whose size scales exponentiall with ``n``
"""
function choimatrix(circuit::Vector{<:Any}; kwargs...)
  return choimatrix(nqubits(circuit), circuit; kwargs...)
end

function choimatrix(N::Int, circuit::Vector{<:Any}; kwargs...)
  return choimatrix(qubits(N), circuit; kwargs...)
end

function choimatrix(
  sites::Vector{<:Index}, circuit::Vector{<:Any}; noise=nothing, kwargs...
)
  return choimatrix(sites, buildcircuit(sites, circuit; noise); kwargs...)
end

function choimatrix(
  sites::Vector{<:Index}, circuit_tensors::Vector{<:Vector{<:ITensor}}; kwargs...
)
  return choimatrix(sites, vcat(circuit_tensors...); kwargs...)
end

function choimatrix(
  sites::Vector{<:Index},
  circuit_tensors::Vector{<:ITensor};
  full_representation=false,
  eltype=nothing,
  device=identity,
  kwargs...,
)
  Λ₀ = unitary_mpo_to_choi_mpo(productoperator(sites))
  Λ₀ = device(_convert_leaf_eltype(eltype, Λ₀))

  inds_sizes = [length(inds(g)) for g in circuit_tensors]
  noiseflag = any(x -> x % 2 == 1, inds_sizes)
  !noiseflag && error("choi matrix requires noise")

  for tensor in circuit_tensors
    addtags!(tensor, "Output")
  end
  # contract to compute the Choi matrix
  Λ = Λ₀
  Λ = full_representation ? convert_to_full_representation(Λ) : Λ
  return runcircuit(Λ, circuit_tensors; apply_dag=true, kwargs...)
end

"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                               EXACT SIMULATOR
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

function runcircuit(
  T::ITensor,
  circuit_tensors::Vector{<:ITensor};
  eltype=nothing,
  device=identity,
  apply_dag=nothing,
  kwargs...,
)
  T = device(_convert_leaf_eltype(eltype, T))
  circuit_tensors = device(_convert_leaf_eltype(eltype, circuit_tensors))

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
      T = is_operator(T) ? T : prime(T, "Site") * dag(T)
      # ρ -> ε(ρ) (MPO -> MPO, conjugate evolution)
      return apply(circuit_tensors, T; apply_dag=true)
      # Pure state evolution
    else
      # |ψ⟩ -> U |ψ⟩ (MPS -> MPS)
      #  ρ  -> U ρ U† (MPO -> MPO, conjugate evolution)
      if !is_operator(T)
        return apply(circuit_tensors, T)
      else
        return apply(circuit_tensors, T; apply_dag=true)
      end
    end
    # Custom mode (apply_dag = true / false)
  else
    Tc = apply(circuit_tensors, T; apply_dag)
    !is_operator(T) && apply_dag && return prime(Tc, "Site") * dag(Tc)
    return Tc
  end
end
