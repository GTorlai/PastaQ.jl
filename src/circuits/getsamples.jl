"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                   MEASUREMENTS / STATE PREPARATION SETTINGS                  -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

@doc raw"""
    fullbases(n::Int; local_basis = "Pauli")

Generate the full set of measurement bases for a choice of local single-qubit basis set.
Predefined option:
+ `local_basis = "Pauli"`: set of ``3^n`` Pauli measurement bases
"""
function fullbases(N::Int; local_basis="Pauli")
  local_basis == "Pauli" && (local_basis = ["X", "Y", "Z"])
  if N > 15
    print(
      "The $(N)-qubit set of Pauli bases contains $(3^N) bases.\n This may take a while...\n\n",
    )
  end
  !(local_basis isa AbstractArray) && error("Basis not recognized")
  A = collect(Iterators.product(ntuple(i -> local_basis, N)...))
  B = reverse.(reshape(A, length(A), 1))
  return reduce(hcat, getindex.(B, i) for i in 1:N)
end

@doc raw"""
    fullpreparations(n::Int; local_input_states = "Pauli")

Generate the full set of ``n``-qubit input states built out of a collection of 
``D=M^n`` states out of ``M`` single-qubit states. Predefined options:
+ `local_input_states = "Pauli"`: ``D=6^n`` Pauli eigenstates
+ `local_input_states = "Tetra"`: ``D=4^n`` 1-qubit SIC-POVM 
"""
function fullpreparations(N::Int; local_input_states="Pauli")
  if N > 5
    print(
      "The $(N)-qubit set of Pauli eigenstates contains $(6^N) bases.\n This may take a while...\n\n",
    )
  end
  local_input_states == "Pauli" &&
    (local_input_states = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"])
  local_input_states == "Tetra" &&
    (local_input_states = ["Tetra1", "Tetra2", "Tetra3", "Tetra4"])
  !(local_input_states isa AbstractArray) && error("States not recognized")
  A = collect(Iterators.product(ntuple(i -> local_input_states, N)...))
  B = reverse.(reshape(A, length(A), 1))
  return reduce(hcat, getindex.(B, i) for i in 1:N)
end

@doc raw"""
    randombases(n::Int, nbases::Int; local_basis = "Pauli")

Generate `nbases` measurement bases composed by ``n`` single-qubit bases. 
By default, each local basis is randomly selected between Pauli bases `["X","Y","Z"]`, with
`"Z"` being the default basis where the quantum state is written.

The measurement bases can also be defined by the user, `local_basis = [O1, O2,...]`, 
as long as the single-qubit Hermitian operators ``O_j`` are defined.  
"""
function randombases(N::Int, nbases::Int; local_basis="Pauli")
  local_basis == "Pauli" && (local_basis = ["X", "Y", "Z"])
  return rand(local_basis, nbases, N)
end

@doc raw"""
    randompreparations(n::Int, npreps::Int;
                       local_input_state = "Pauli")

Generate `npreps` random input states to a quantum circuit. Each ``n``-qubit
input state is selected according to the following options:
+ `local_input_states = "Pauli"`: ``D=6^n`` Pauli eigenstates
+ `local_input_states = "Tetra"`: ``D=4^n`` 1-qubit SIC-POVM 

The input states can also be set to a user-defined set, 
`local_input_states = ["A","B","C",...]`, assuming single-qubit states ``|A\rangle``, 
``|B\rangle`` have been properly defined.
"""
function randompreparations(N::Int, npreps::Int; local_input_states="Pauli")
  local_input_states == "Pauli" &&
    (local_input_states = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"])
  local_input_states == "Tetra" && (local_input_states = ["T1", "T2", "T3", "T4"])
  # One shot per basis
  return rand(local_input_states, npreps, N)
end

"""
    measurementgates(basis::Vector)

Given as input a measurement basis, returns the corresponding
gate data structure. If the basis is `"Z"`, no action is required.
If not, a quantum gate corresponding to the given basis rotation
is added to the list.

Example:
  basis = `["X","Z","Z","Y"]`

  -> `gate_list = [("basisX", 1),("basisY", 4)]`
"""
function measurementgates(basis::Vector)
  gate_list = Tuple[]
  for j in 1:length(basis)
    if basis[j] ≠ "Z"
      push!(gate_list, ("basis$(basis[j])", j, (adjoint=true,)))
    end
  end
  return gate_list
end

"""
    readouterror!(measurement::Union{Vector, Matrix}, p1given0, p0given1)

Add readout error to a single projective measurement.

# Arguments:
  - `measurement`: bit string of projective measurement outcome
  - `p1given0`: readout error probability 0 -> 1
  - `p0given1`: readout error probability 1 -> 0
"""
function readouterror!(
  measurement::Union{Vector,Matrix}, p1given0::Float64, p0given1::Float64
)
  for j in 1:size(measurement)[1]
    if measurement[j] == 0
      measurement[j] = StatsBase.sample([0, 1], Weights([1 - p1given0, p1given0]))
    else
      measurement[j] = StatsBase.sample([0, 1], Weights([p0given1, 1 - p0given1]))
    end
  end
  return measurement
end

"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                            PROJECTIVE MEASUREMENTS                           -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

@doc raw"""
    getsamples(M::Union{MPS,MPO}, nshots::Int; kwargs...)
    getsamples(T::ITensor, nshots::Int)

Perform `nshots` projective measurements of a wavefunction 
``|\psi\rangle`` or density operator ``\rho`` in the MPS/MPO reference basis. 
Each measurement consists of a binary vector ``\sigma = (\sigma_1,\sigma_2,\dots)``, 
drawn from the probabilty distributions:
- ``P(\sigma) = |\langle\sigma|\psi\rangle|^2``,   if ``M = |\psi\rangle`` is an `MPS`.
- ``P(\sigma) = \langle\sigma|\rho|\sigma\rangle``   :  if ``M = \rho`` is an `MPO`.
"""
function getsamples(M0::Union{MPS,MPO}, nshots::Int; kwargs...)
  nthreads = Threads.nthreads()
  data = [Vector{Vector{Int64}}(undef, 0) for _ in 1:nthreads]
  M = orthogonalize!(copy(M0), 1)

  Threads.@threads for j in 1:nshots
    nthread = Threads.threadid()
    sample_ = getsamples!(M; kwargs...)
    push!(data[nthread], sample_)
  end
  return permutedims(hcat(vcat(data...)...))
end

@doc raw"""
    getsamples(M::Union{MPS,MPO,ITensor}, bases::Matrix::Matrix{<:String}, nshots::int; kwargs...)
    getsamples(M::Union{MPS,MPO,ITensor}, bases::Vector{<:Vector}, nshots::Int; kwargs...) = 

Generate a set of measurements acccording to a set
of input `bases`, performing `nshots` measurements per basis. 
For a single measurement, a depth-1 unitary ``U`` is applied to the input
state ``M`` according to the `basis`. The probability of recording outcome
``\sigma = (\sigma_1,\sigma_2,\dots)`` in the basis defined by ``U`` is
- ``P(\sigma) = |\langle\sigma|U\psi\rangle|^2``,   if ``M = |\psi\rangle`` is an `MPS`.
- ``P(\sigma) = \langle\sigma|U\rho U^\dagger|\sigma\rangle``,   if ``M = \rho`` is an `MPO`.
"""
function getsamples(
  M0::Union{MPS,MPO,ITensor}, bases::Matrix{<:String}, nshots::Int; kwargs...
)
  N = nsites(M0)
  @assert N == size(bases)[2]
  nthreads = Threads.nthreads()
  data = [Vector{Vector{Pair{String,Int}}}(undef, 0) for _ in 1:nthreads]
  M = copy(M0)
  !(M isa ITensor) && orthogonalize!(M, 1)

  Threads.@threads for n in 1:size(bases, 1)
    nthread = Threads.threadid()
    meas_gates = measurementgates(bases[n, :])
    M_meas = runcircuit(M, meas_gates)
    measurements = getsamples(M_meas, nshots; kwargs...)
    basisdata = [[bases[n, j] => measurements[k, j] for j in 1:N] for k in 1:nshots]
    append!(data[nthread], basisdata)
  end
  return permutedims(hcat(vcat(data...)...))
end

function getsamples(
  M::Union{MPS,MPO,ITensor}, bases::Vector{<:Vector}, nshots::Int; kwargs...
)
  return getsamples(M, permutedims(hcat(bases...)), nshots; kwargs...)
end

function getsamples(
  M::Union{MPS,MPO,ITensor}, bases::Union{Matrix,Vector{<:Vector}}; kwargs...
)
  return getsamples(M, bases, 1; kwargs...)
end

function getsamples(
  T::ITensor, nshots::Int; readout_errors=(p1given0=nothing, p0given1=nothing)
)
  p1given0 = readout_errors[:p1given0]
  p0given1 = readout_errors[:p0given1]

  # Get the number of qubits
  N = nsites(T)

  # Get a dense array which can be
  # - Vector with dim 2^N for a wavefunction |ψ⟩
  # - Matrix with dim (2^N,2^N) for a density matrix ρ
  A = array(T)

  # Compute the full probability distribution 
  # P(σ) = |⟨σ|ψ⟩|² (Tr[ρ|σ⟩⟨σ|] 
  probs = (A isa Vector ? abs2.(A) : real(diag(A)))
  @assert sum(probs) ≈ 1

  # Sample the distribution exactly
  index = StatsBase.sample(0:(1 << N - 1), StatsBase.Weights(probs), nshots)

  # Map integer to binary vectors and massage the structure
  M = hcat(digits.(index, base=2, pad=N)...)'
  measurements = reverse(M; dims=2)
  if !isnothing(p1given0) || !isnothing(p0given1)
    p1given0 = (isnothing(p1given0) ? 0.0 : p1given0)
    p0given1 = (isnothing(p0given1) ? 0.0 : p0given1)
    for n in 1:nshots
      readouterror!(measurements[n, :], p1given0, p0given1)
    end
  end
  return measurements
end

getsamples(T::ITensor; kwargs...) = vcat(getsamples!(T, 1; kwargs...)...)

"""
    getsamples!(M::Union{MPS,MPO})

Generate a single projective measurement in the MPS/MPO reference basis.
If `readout_errors` is non-trivial, readout errors with given probabilities
are applied to the measurement outcome.
"""
function getsamples!(M::Union{MPS,MPO}; readout_errors=(p1given0=nothing, p0given1=nothing))
  p1given0 = readout_errors[:p1given0]
  p0given1 = readout_errors[:p0given1]
  orthogonalize!(M, 1)
  measurement = sample(M)
  measurement .-= 1
  if !isnothing(p1given0) || !isnothing(p0given1)
    p1given0 = (isnothing(p1given0) ? 0.0 : p1given0)
    p0given1 = (isnothing(p0given1) ? 0.0 : p0given1)
    readouterror!(measurement, p1given0, p0given1)
  end
  return measurement
end

"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                            PROCESS TOMOGRAPHY DATA                                -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

@doc raw"""
    getsamples(
      M::Union{LPDO,MPO,ITensor},
      preps::Matrix,
      bases::Matrix,
      nshots::Int;
      kwargs...
    )

Generate a set of process measurement data acccording to a set
of input states `preps` and measurement `bases`, performing `nshots` measurements per configuration. 
For a single measurement, the input state ``|\xi\rangle=\otimes_j|\xi_j\rangle`` is evolved according to the channel ``M``,
and then measured in a given measurement basis. The probability that the final state returns outcome
``\sigma = (\sigma_1,\sigma_2,\dots)`` in the basis defined by ``U`` is given by 

```math
P(\sigma|\xi) = \text{Tr}\big[(\rho_\xi^T\otimes 1)\Lambda_{M} \big]
```
where ``\rho_\xi = |\xi\rangle\langle\xi|`` and ``\Lambda_M`` is the Choi matrix.
"""
function getsamples(
  M0::Union{LPDO,MPO,ITensor}, preps::Matrix, bases::Matrix, nshots::Int; kwargs...
)
  N = nsites(M0)
  npreps = size(preps, 1)
  nbases = size(bases, 1)
  ntotal = npreps * nbases

  preps_and_bases = Matrix{Pair{String,String}}(undef, 0, N)
  for p in 1:npreps
    x = [preps[p, :] .=> bases[b, :] for b in 1:nbases]
    preps_and_bases = vcat(preps_and_bases, permutedims(hcat(x...)))
  end
  return getsamples(M0, preps_and_bases, nshots; kwargs...)
end

"""
    projectchannel(U0::MPO,prep::Array)
    projectchannel(U::ITensor, prep::Array)

Project the unitary circuit (MPO) into a state `prep` 
made out of single-qubit Pauli eigenstates (e.g. `|ϕ⟩ =|+⟩⊗|0⟩⊗|r⟩⊗…).
The resulting MPS describes the quantum state obtained by applying
the quantum circuit to `|ϕ⟩`. Same for a dense ITensors.
"""
function _projectchannel(U0::Union{MPO,ITensor}, prep::Array)
  U = copy(U0)
  s = U isa MPO ? firstsiteinds(U) : inds(U; plev=0)
  for j in 1:length(s)
    if U isa MPO
      U[j] = U[j] * state(prep[j], s[j])
    else
      U = U * state(prep[j], s[j])
    end
  end
  U isa ITensor && return noprime!(U)
  return convert(MPS, noprime!(U))
end

function _projectchannel(Λ0::Choi, prep::Array)
  Λ = copy(Λ0.X)
  s = Λ isa MPO ? firstsiteinds(Λ; tags="Input") : inds(Λ; tags="Input", plev=0)
  for j in 1:length(s)
    # No conjugate on the gate (transpose input!)
    if Λ isa MPO
      Λ[j] = Λ[j] * dag(state(prep[j], s[j]))
      Λ[j] = Λ[j] * prime(state(prep[j], s[j]))
    else
      Λ = Λ * dag(state(prep[j], s[j]))
      Λ = Λ * prime(state(prep[j], s[j]))
    end
  end
  return Λ
end

function projectchannel(M::Union{MPO,ITensor}, prep::AbstractArray)
  return ischoi(M) ? _projectchannel(Choi(M), prep) : _projectchannel(M, prep)
end

function getsamples(
  M0::Union{LPDO,MPO,ITensor}, preps_and_bases::Matrix{<:Pair}, nshots::Int, kwargs...
)
  N = nsites(M0)
  @assert N == size(preps_and_bases, 2)

  nthreads = Threads.nthreads()
  data = [Matrix{Pair{String,Pair{String,Int}}}(undef, 0, N) for _ in 1:nthreads]
  M = copy(M0)

  Threads.@threads for k in 1:size(preps_and_bases, 1)
    nthread = Threads.threadid()
    # input state to the channel
    prep = first.(preps_and_bases[k, :])
    # measurement basis
    basis = last.(preps_and_bases[k, :])

    # project the channel into the input state
    Mproj = projectchannel(M, prep)

    # perform measurement basis rotation
    meas_gates = measurementgates(basis)
    M_meas = runcircuit(Mproj, meas_gates)

    # perform projective measurement
    measurements = getsamples(M_meas, nshots; kwargs...)
    basisdata = [
      [prep[j] => (basis[j] => measurements[k, j]) for j in 1:N] for k in 1:nshots
    ]
    data[nthread] = vcat(data[nthread], permutedims(hcat(basisdata...)))
  end
  return vcat(data...)
end

function getsamples(
  M::Union{LPDO,MPO,ITensor},
  preps::Vector{<:Vector},
  bases::Vector{<:Vector},
  nshots::Int;
  kwargs...,
)
  return getsamples(
    M, permutedims(hcat(preps...)), permutedims(hcat(bases...)), nshots; kwargs...
  )
end

function getsamples(
  M::Union{LPDO,MPO,ITensor},
  preps::Union{Matrix,Vector{<:Vector}},
  bases::Union{Matrix,Vector{<:Vector}};
  kwargs...,
)
  return getsamples(M, preps, bases, 1; kwargs...)
end
