"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
-                   MEASUREMENTS / STATE PREPARATION SETTINGS                  -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""

"""
    fullbases(N::Int; local_basis = "Pauli")

Generate the full set of bases for a choice of local single-qubit basis set.
Predefined option: "Pauli", # bases = 3^N
"""
function fullbases(N::Int; local_basis = "Pauli")
  local_basis == "Pauli" && (local_basis = ["X","Y","Z"]) 
  if N >15
    print("The $(N)-qubit set of Pauli bases contains $(3^N) bases.\n This may take a while...\n\n")
  end
  !(local_basis isa AbstractArray) && error("Basis not recognized")
  A = Iterators.product(ntuple(i->local_basis, N)...) |> collect
  B = reverse.(reshape(A,length(A),1))
  return  reduce(hcat, getindex.(B,i) for i in 1:N)
end

"""
    fullpreparations(N::Int; local_input_states="Pauli")

Generate the full set of input states to a channel. 
Predefined option: "Pauli", 6^N
                   "Tetra", 4^N
"""
function fullpreparations(N::Int; local_input_states="Pauli")
  if N > 5
    print("The $(N)-qubit set of Pauli eigenstates contains $(6^N) bases.\n This may take a while...\n\n")
  end
  local_input_states == "Pauli" && (local_input_states = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"])
  local_input_states == "Tetra" && (local_input_states = ["SIC1","SIC2","SIC3","SIC4"]) 
  !(local_input_states isa AbstractArray) && error("States not recognized")
  A = Iterators.product(ntuple(i->local_input_states, N)...) |> collect
  B = reverse.(reshape(A,length(A),1))
  return  reduce(hcat, getindex.(B,i) for i in 1:N)
end

"""
    randombases(N::Int, nbases::Int; local_basis = "Pauli")

Generate `nbases` measurement bases. By default, each
local basis is randomly selected between `["X","Y","Z"]`, with
`"Z"` being the default basis where the quantum state is written.
"""
function randombases(N::Int, nbases::Int; local_basis = "Pauli")
  local_basis == "Pauli" && (local_basis = ["X","Y","Z"]) 
  return rand(local_basis, nbases, N)
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
      push!(gate_list, ("basis$(basis[j])", j, (dag=true,)))
    end
  end
  return gate_list
end

"""
    randompreparations(N::Int, nshots::Int;
                       local_input_state = ["X+","X-","Y+","Y-","Z+","Z-"],
                       ndistinctstates = nothing)

Generate `nshots` input states to a quantum circuit. By default, each
single-qubit state is randomly selected between the 6 eigenstates of
the Pauli matrices, `["X+","X-","Y+","Y-","Z+","Z-"]`.
If `ndistinctstates` is provided, the output consist of `numprep`
different input states, each being repeated `nshots÷ndistinctstates`
times.
"""
function randompreparations(
  N::Int,
  npreps::Int;
  local_input_states= "Pauli", 
)
 
  local_input_states == "Pauli" && (local_input_states = ["X+", "X-", "Y+", "Y-", "Z+", "Z-"])
  local_input_states == "Tetra" && (local_input_states = ["SIC1","SIC2","SIC3","SIC4"]) 
  # One shot per basis
  return rand(local_input_states, npreps, N)
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

"""
    getsamples(T::ITensor, nshots::Int)

Generate `nshots` projective measurements for an input quantum state `T`,
which can either be a wavefunction or density matrix (dense).
"""
function getsamples(T::ITensor, nshots::Int; 
    readout_errors = (p1given0 = nothing, p0given1 = nothing))
  
  p1given0 = readout_errors[:p1given0]
  p0given1 = readout_errors[:p0given1]
 
  # Get the number of qubits
  N = length(T)
  
  # Get a dense array which can be
  # - Vector with dim 2^N for a wavefunction |ψ⟩
  # - Matrix with dim (2^N,2^N) for a density matrix ρ
  A = array(T)
  
  # Compute the full probability distribution 
  # P(σ) = |⟨σ|ψ⟩|² (Tr[ρ|σ⟩⟨σ|] 
  probs = (A isa Vector ? abs2.(A) : real(diag(A)))
  @assert sum(probs) ≈ 1
  
  # Sample the distribution exactly
  index = StatsBase.sample(0:1<<N-1, StatsBase.Weights(probs), nshots)
  
  # Map integer to binary vectors and massage the structure
  M = hcat(digits.(index,base=2,pad=N)...)'
  measurements = reverse(M,dims=2)
  if !isnothing(p1given0) || !isnothing(p0given1)
    p1given0 = (isnothing(p1given0) ? 0.0 : p1given0)
    p0given1 = (isnothing(p0given1) ? 0.0 : p0given1)
    for n in 1:nshots
      readouterror!(measurements[n,:], p1given0, p0given1)
    end
  end
  return measurements
end

getsamples(T::ITensor; kwargs...) = 
  vcat(getsamples!(T,1; kwargs...)...)


"""
    getsamples!(M::Union{MPS,MPO};
                readout_errors = (p1given0 = nothing,
                                  p0given1 = nothing))

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
    PastaQ.getsamples!(M::Union{MPS,MPO}, nshots::Int; kwargs...)

Perform `nshots` projective measurements of a wavefunction 
`|ψ⟩` or density operator `ρ` in the MPS/MPO reference basis. 
Each measurement consists of a binary vector `σ = (σ₁,σ₂,…)`, 
drawn from the probabilty distribution:
- `P(σ) = |⟨σ|ψ⟩|²`  :  if `M = ψ is MPS`
- `P(σ) = ⟨σ|ρ|σ⟩`   :  if `M = ρ is MPO`
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

"""
    getsamples(M::Union{MPS,MPO,ITensor}, bases::Matrix; kwargs...)

Generate a dataset of measurements acccording to a set
of input `bases`, performing `nshots` measurements per basis. 
For a single measurement, `Û` is the depth-1 
local circuit rotating each qubit, the  data-point `σ = (σ₁,σ₂,…)`
is drawn from the probability distribution:
- `P(σ) = |⟨σ|Û|ψ⟩|²`    :  if `M = ψ is MPS` 
- `P(σ) = <σ|Û ρ Û†|σ⟩`  :  if `M = ρ is MPO`   
"""
function getsamples(M0::Union{MPS,MPO,ITensor}, bases::Matrix, nshots::Int; kwargs...)
  N = length(M0)
  @assert N == size(bases)[2]
  nthreads = Threads.nthreads()
  data = [Vector{Vector{Pair{String,Int}}}(undef, 0) for _ in 1:nthreads]
  M = copy(M0)
  !(M isa ITensor) && orthogonalize!(M, 1)
  
  Threads.@threads for n in 1:size(bases, 1)
    nthread = Threads.threadid()
    meas_gates = measurementgates(bases[n, :])
    M_meas = runcircuit(copy(M), meas_gates)
    measurements = getsamples(M_meas, nshots; kwargs...)
    basisdata = [[bases[n,j] => measurements[k,j] for j in 1:N] for k in 1:nshots]
    append!(data[nthread], basisdata)
  end
  return permutedims(hcat(vcat(data...)...))
end

getsamples(M::Union{MPS,MPO,ITensor}, bases::Vector{<:Vector}, nshots::Int; kwargs...) = 
  getsamples(M, permutedims(hcat(bases...)), nshots; kwargs...)

getsamples(M::Union{MPS,MPO,ITensor}, bases::Union{Matrix,Vector{<:Vector}}; kwargs...) = 
  getsamples(M, bases, 1; kwargs...)




"""
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
                            PROCESS TOMOGRAPHY DATA                                -
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
"""


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
  s = U isa MPO ? firstsiteinds(U) : inds(U; plev = 0)
  for j in 1:length(s)
    if U isa MPO
      U[j] = U[j] * state(prep[j], s[j])
    else
      U = U * state(prep[j], s[j]) 
    end
  end
  U isa ITensor && return noprime!(U)
  return convert(MPS,noprime!(U))
end

function _projectchannel(Λ0::Choi, prep::Array)
  Λ = copy(Λ0.X)
  s = Λ isa MPO ? firstsiteinds(Λ; tags="Input") : inds(Λ; tags = "Input", plev = 0)
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

projectchannel(M::Union{MPO,ITensor}, prep::AbstractArray) = 
  ischoi(M) ? _projectchannel(Choi(M), prep) : _projectchannel(M, prep) 


function getsamples(
  M0::Union{LPDO,MPO,ITensor},
  preps::Matrix,
  bases::Matrix,
  nshots::Int;
  readout_errors=(p1given0=nothing, p0given1=nothing),
)
  N = length(M0)
  @assert N == size(preps, 2)

  nthreads = Threads.nthreads()
  data = [Matrix{Pair{String,Pair{String,Int}}}(undef, 0, N) for _ in 1:nthreads]
  M = copy(M0)
  Threads.@threads for p in 1:size(preps, 1)
    nthread = Threads.threadid()
    Mproj = projectchannel(M, preps[p,:])
    prepdata = permutedims(hcat(repeat([preps[p,:]], nshots * size(bases,1))...))
    basisdata = getsamples(Mproj, bases, nshots; readout_errors=readout_errors)
    data[nthread] = vcat(data[nthread], prepdata .=> basisdata)
  end
  return vcat(data...)
end

getsamples(M::Union{LPDO,MPO, ITensor}, preps::Vector{<:Vector}, bases::Vector{<:Vector}, nshots::Int; kwargs...) = 
  getsamples(M, permutedims(hcat(preps...)), permutedims(hcat(bases...)), nshots; kwargs...)

getsamples(M::Union{LPDO,MPO, ITensor}, preps::Union{Matrix,Vector{<:Vector}}, bases::Union{Matrix,Vector{<:Vector}}; kwargs...) = 
  getsamples(M, preps, bases, 1; kwargs...)


