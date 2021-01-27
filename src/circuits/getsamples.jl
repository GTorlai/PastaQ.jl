"""
    randombases(N::Int, nshots::Int;
                local_basis = ["X","Y","Z"],
                ndistinctbases = nothing)

Generate `nshots` measurement bases. By default, each
local basis is randomly selected between `["X","Y","Z"]`, with
`"Z"` being the default basis where the quantum state is written.
If `ndistinctbases` is provided, the output consist of `ndistinctbases`
different measurement bases, each being repeated `nshots÷ndistinctbases`
times.
"""
function randombases(N::Int, numshots::Int;
                     local_basis = ["X","Y","Z"],
                     ndistinctbases = nothing)
  # One shot per basis
  if isnothing(ndistinctbases)
    bases = rand(local_basis,numshots,N)
  # Some number of shots per basis
  else
    @assert(numshots%ndistinctbases ==0)
    shotsperbasis = numshots÷ndistinctbases
    bases = repeat(rand(local_basis,1,N),shotsperbasis)
    for n in 1:ndistinctbases-1
      newbases = repeat(rand(local_basis,1,N),shotsperbasis)
      bases = vcat(bases,newbases)
    end
  end
  return bases
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
      push!(gate_list, ("basis$(basis[j])", j, (dag = true,)))
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
function randompreparations(N::Int, nshots::Int;
                            local_input_state = ["X+","X-","Y+","Y-","Z+","Z-"],
                            ndistinctstates = nothing)
  # One shot per basis
  if isnothing(ndistinctstates)
    preparations = rand(local_input_state,nshots,N)
  else
    @assert(nshots%ndistinctstates == 0 )
    shotsperstate = nshots÷ndistinctstates
    preparations = repeat(rand(local_input_state,1,N),shotsperstate)
    for n in 1:ndistinctstates-1
      newstates = repeat(rand(local_input_state,1,N),shotsperstate)
      preparations = vcat(preparations,newstates)
    end
  end
  return preparations
end


"""
    readouterror!(measurement::Union{Vector, Matrix}, p1given0, p0given1)

Add readout error to a single projective measurement.

# Arguments:
  - `measurement`: bit string of projective measurement outcome
  - `p1given0`: readout error probability 0 -> 1
  - `p0given1`: readout error probability 1 -> 0
"""
function readouterror!(measurement::Union{Vector, Matrix},
                       p1given0::Float64,
                       p0given1::Float64)

  for j in 1:size(measurement)[1]
    if measurement[j] == 0
      measurement[j] = StatsBase.sample([0,1],Weights([1-p1given0,p1given0]))
    else
      measurement[j] = StatsBase.sample([0,1],Weights([p0given1,1-p0given1]))
    end
  end
  return measurement
end

"""
    getsamples!(M::Union{MPS,MPO};
                readout_errors = (p1given0 = nothing,
                                  p0given1 = nothing))

Generate a single projective measurement in the MPS/MPO reference basis.
If `readout_errors` is non-trivial, readout errors with given probabilities
are applied to the measurement outcome.
"""
function getsamples!(M::Union{MPS,MPO};
                     readout_errors = (p1given0 = nothing,
                                       p0given1 = nothing))
  p1given0 = readout_errors[:p1given0]
  p0given1 = readout_errors[:p0given1]
  orthogonalize!(M,1)
  measurement = sample(M)
  measurement .-= 1
  if !isnothing(p1given0) || !isnothing(p0given1)
    p1given0 = (isnothing(p1given0) ? 0.0 : p1given0)
    p0given1 = (isnothing(p0given1) ? 0.0 : p0given1)
    readouterror!(measurement,p1given0,p0given1)
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
function getsamples!(M::Union{MPS,MPO},nshots::Int; kwargs...)
  measurements = Matrix{Int64}(undef, nshots, length(M))
  for n in 1:nshots
    measurements[n,:] = getsamples!(M; kwargs...)
  end
  return measurements
end

#
# MEASUREMENT IN MULTIPLE BASES
#

"""
    getsamples(M::Union{MPS,MPO}, bases::Array)

Generate a dataset of measurements acccording to a set
of input `bases`. For a single measurement, `Û` is the depth-1 
local circuit rotating each qubit, the  data-point `σ = (σ₁,σ₂,…)`
is drawn from the probability distribution:
- `P(σ) = |⟨σ|Û|ψ⟩|²`    :  if `M = ψ is MPS` 
- `P(σ) = <σ|Û ρ Û†|σ⟩`  :  if `M = ρ is MPO`   
"""
function getsamples(M0::Union{MPS,MPO}, bases::Array; kwargs...)
  @assert length(M0) == size(bases)[2]
  data = Matrix{Pair{String, Int}}(undef, size(bases)[1],length(M0))
  M = copy(M0)
  orthogonalize!(M,1)
  for n in 1:size(bases)[1]
    meas_gates = measurementgates(bases[n,:])
    M_meas = runcircuit(M,meas_gates)
    measurement = getsamples!(M_meas;kwargs...)
    data[n,:] .= bases[n,:] .=> measurement
  end
  return data 
end

"""
    getsamples(M::Union{MPS,MPO}, nshots::Int;
               local_basis = ["X", "Y", "Z"],
               ndistinctbases = nothing)

Perform `nshots` projective measurements of a wavefunction 
`|ψ⟩` or density operator `ρ`. The measurement consists of
a binary vector `σ = (σ₁,σ₂,…)`, drawn from the probabilty
distribution:
- `P(σ) = |⟨σ|Û|ψ⟩|²`    :  if `M = ψ is MPS`
- `P(σ) = ⟨σ|Û ρ Û†|σ⟩`  :  if `M = ρ is MPO`

For a single measurement, `Û` is the depth-1
local circuit rotating each qubit, where the rotations are determined by
randomly choosing from the bases specified by `local_basis`
keyword argument (i.e. if `"X"` is chosen out of `["X", "Y", "Z"]`,
the rotation is the eigenbasis of `"X"`).
"""
function getsamples(M::Union{MPS,MPO}, nshots::Int64;
                    local_basis = nothing,
                    ndistinctbases = nothing,
                    readout_errors = (p1given0 = nothing,
                                      p0given1 = nothing))
  if isnothing(local_basis)
    data = getsamples!(copy(M), nshots; readout_errors = readout_errors)
  else
    bases = randombases(length(M), nshots;
                        local_basis = local_basis,
                        ndistinctbases = ndistinctbases)
    data = getsamples(M, bases; readout_errors = readout_errors)
  end
  return data
end

#
# QUANTUM PROCESS TOMOGRAPHY
#

""" 
    getsamples(M0::Union{MPS, MPO},
               gate_tensors::Vector{<: ITensor},
               prep::Array,
               basis::Array;
               cutoff::Float64 = 1e-15,
               maxdim::Int64 = 10000,
               kwargs...)

Generate a single data-point for quantum process tomography, 
consisting of an input state (a product state of single-qubit
Pauli eigenstates) and an output state measured after a given
basis rotation is performed at the output of a quantum channel.

# Arguments:
 - `M0`: reference input state (to avoid re-compiling the circuit)
 - `gate_tensors`: tensors for the channel (unitary or noisy)
 - `prep`: a prepared input state (e.g. `["X+","Z-","Y+","X-"]`)
 - `basis`: a measuremement basis (e.g. `["Z","Z","Y","X"])
"""

function getsamples(hilbert0::Vector{<:Index},
                    gate_tensors::Vector{<:ITensor},
                    prep::Array, basis::Array;
                    cutoff::Float64 = 1e-15,
                    maxdim::Int64 = 10000,
                    readout_errors = nothing,
                    kwargs...)
  # Generate preparation/measurement gates
  meas_gates = measurementgates(basis)
  # Prepare quantum state
  M_in = qubits(hilbert0, prep)

  # TODO: delete
  #M0 = qubits(hilbert0)
  #prep_gates = preparationgates(prep)
  #M_in  = runcircuit(M0, prep_gates)

  # Apply the quantum channel
  M_out = runcircuit(M_in, gate_tensors,
                     cutoff = cutoff, maxdim = maxdim) 
  # Apply basis rotation
  M_meas = runcircuit(M_out, meas_gates)
  # Measure
  measurement = getsamples!(M_meas; readout_errors = readout_errors)
  
  return basis .=> measurement
end


"""
    projectchoi(Λ0::MPO, prep::Array)

Project the Choi matrix  input indices into a state `prep` 
made out of single-qubit Pauli eigenstates (e.g. `|ϕ⟩ =|+⟩⊗|0⟩⊗|r⟩⊗…).
The resulting MPO describes the quantum state obtained by applying
the quantum channel underlying the Choi matrix to `|ϕ⟩`.
"""
function projectchoi(Λ0::MPO, prep::Array)
  Λ = copy(Λ0)
  #choi = Λ.M
  #st = "state" .* copy(prep) 
  st = prep
  s = firstsiteinds(Λ, tags="Input")
  
  for j in 1:length(Λ)
    # No conjugate on the gate (transpose input!)
    Λ[j] = Λ[j] * dag(state(st[j],s[j]))
    Λ[j] = Λ[j] * prime(state(st[j],s[j]))
  end
  return Λ
end


"""
    projectunitary(U0::MPO,prep::Array)

Project the unitary circuit (MPO) into a state `prep` 
made out of single-qubit Pauli eigenstates (e.g. `|ϕ⟩ =|+⟩⊗|0⟩⊗|r⟩⊗…).
The resulting MPS describes the quantum state obtained by applying
the quantum circuit to `|ϕ⟩`.
"""
function projectunitary(U::MPO,prep::Array)
  #st = "state" .* copy(prep) 
  st = prep
  M = ITensor[]
  s = firstsiteinds(U)
  for j in 1:length(U)
    push!(M,U[j] * state(st[j],s[j]))
  end
  return noprime!(MPS(M))
end


"""
    getsamples(N::Int64, gates::Vector{<:Tuple}, nshots::Int64;     
               noise = nothing,
               process::Bool = false,              
               build_process::Bool = false,
               local_basis::Array = ["X","Y","Z"],                   
               local_input_state::Array = ["X+","X-","Y+","Y-","Z+","Z-"],
               ndistinctbases = nothing,
               ndistinctstates = nothing,
               cutoff::Float64 = 1e-15,
               maxdim::Int64 = 10000,         
               kwargs...)                                         

Generate `nshots` data-point for quantum state tomography or quantum process tomography for a 
quantum channel corresponding to a set of quantum `gates` and a `noise` model. 

# Arguments:
  - `gates`: a set of quantum gates
  - `noise`: apply a noise model after each quantum gate in the circuit
  - `process`: if false, generate data for state tomography, where the state is defined by the gates applied to the state `|0,0,...,⟩`. If true, generate data for process tomography.
  - `build_process`: if true, generate data by building the full unitary circuit or Choi matrix, and then sampling from that unitary circuit or Choi matrix (as opposed to running the circuit many times on different initial states). It is only used if `process = true`.
  - `local_input_state`: a set of input states (e.g. `["X+","X-","Y+","Y-","Z+","Z-"]`) which are sampled randomly to generate input states.
  - `local_basis`: the local bases (e.g. `["X","Y","Z"]`) which are sampled randomly to perform measurements in a random basis.
"""
function getsamples(N::Int64, gates::Vector{<:Tuple}, nshots::Int64;
                    noise = nothing,
                    build_process::Bool = true,
                    process::Bool = false,
                    local_basis = ["X", "Y", "Z"],
                    local_input_state = ["X+","X-","Y+","Y-","Z+","Z-"],
                    ndistinctbases = nothing,
                    ndistinctstates = nothing,
                    cutoff::Float64 = 1e-15,
                    maxdim::Int64 = 10000,
                    readout_errors = (p1given0 = nothing, p0given1 = nothing),
                    kwargs...)
  
  # Generate data on the quantum state at the output of the channel/circuit
  if !process
    # Apply the quantum channel
    M = runcircuit(N, gates; process = false, noise = noise,
                   cutoff = cutoff, maxdim = maxdim, kwargs...)
    
    # Generate projective measurements
    data = getsamples(M,nshots; 
                      local_basis = local_basis, 
                      ndistinctbases = ndistinctbases,
                      readout_errors = readout_errors)
    return data, M
                      
  else
    
    local_basis = (isnothing(local_basis) ? ["X","Y","Z"] : local_basis)
    
    bases = randombases(N, nshots;
                        local_basis = local_basis,
                        ndistinctbases = ndistinctbases)
    
    preps = randompreparations(N, nshots, local_input_state = local_input_state,
                               ndistinctstates = ndistinctstates)
    
    # Generate the unitary MPO / Choi matrix, then sample from it
    if build_process
      M = runcircuit(N, gates; process = true, noise = noise, 
                     cutoff = cutoff,maxdim = maxdim, kwargs...)
      data = getsamples(M, preps, bases; readout_errors = readout_errors)
      return data, M
    
    # Generate data with full state evolution
    else
      data = getsamples(gates,preps,bases; noise = noise, cutoff = cutoff, maxdim = maxdim,
                        readout_errors = readout_errors, kwargs...)
      return data, nothing
    end
  end
end
getsamples(N::Int64, gates::Vector{Vector{<:Tuple}}, nshots::Int64; kwargs...) = 
  getsamples(N, vcat(gates...), nshots; kwargs...)

getsamples(gates::Vector, nshots::Int64; kwargs...) =
  getsamples(numberofqubits(gates), gates, nshots; kwargs...)


function getsamples(gates::Array,preps::Array, bases::Array ;
                    noise = nothing,cutoff::Float64 = 1e-15,maxdim::Int64 = 10000,
                    readout_errors = (p1given0 = nothing, p0given1 = nothing),
                    kwargs...)
  @assert size(preps) == size(bases)
  N = size(preps)[2]
  nshots = size(preps)[1]
  
  ψ0 = qubits(N)
  hilbert = hilbertspace(ψ0) 
  # Pre-compile quantum channel
  gate_tensors = buildcircuit(ψ0, gates; noise=noise, kwargs...)
  
  data = Matrix{Pair{String, Int}}(undef,nshots,length(ψ0))
  for n in 1:nshots
    data[n,:] = getsamples(hilbert, gate_tensors, preps[n,:], bases[n,:];
                           noise = noise, cutoff = cutoff, maxdim = maxdim,
                           readout_errors = readout_errors, kwargs...)
  end
  return preps .=> data
end


function getsamples(M::Union{LPDO,MPO}, preps::Array, bases::Array;
                    readout_errors = (p1given0 = nothing, p0given1 = nothing))
  
  @assert size(preps) == size(bases)
  nshots = size(preps)[1]
  data = Matrix{Pair{String, Int}}(undef,nshots,length(M))
  # Get unitary MPO / Choi matrix
  for n in 1:nshots
    #M′= (M isa Choi ? projectchoi(M,preps[n,:]) : projectunitary(M,preps[n,:]))
    M′= (ischoi(M) ? projectchoi(M,preps[n,:]) : projectunitary(M,preps[n,:]))
    meas_gates = measurementgates(bases[n,:])
    M_meas = runcircuit(M′,meas_gates)
    measurement = getsamples!(M_meas; readout_errors = readout_errors)
    data[n,:] .= bases[n,:] .=> measurement
    #data[n,:] =  convertdatapoint(measurement,bases[n,:])
  end
  return preps .=> data 
end


