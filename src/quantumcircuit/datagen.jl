"""
  generatedata(M::Union{MPS,MPO},nshots::Int)

Perform `nshots` projective measurements on a wavefunction 
`|ψ⟩` or density operator `ρ`. Each measurement consist of
a binary vector `σ = (σ₁,σ₂,…)`, drawn from the probabilty
distribution:
- P(σ) = |⟨σ|ψ⟩|² : if `M = ψ is MPS`
- P(σ) = ⟨σ|ρ|σ⟩  : if `M = ρ is MPO`
"""

function generatedata(M::Union{MPS,MPO},nshots::Int)
  orthogonalize!(M,1)
  if (nshots==1)
    measurements = sample(M)
    measurements .-= 1
  else
    measurements = Matrix{Int64}(undef, nshots, length(M))
    for n in 1:nshots
      measurement = sample(M)
      measurement .-= 1
      measurements[n,:] = measurement
    end
  end
  return measurements
end

"""
  measurementsettings(N::Int,numshots::Int;
                      bases_id::Array=["X","Y","Z"],
                      numbases=nothing)

Generate `nshots` measurement bases. By default, each
local basis is randomly selected between `["X","Y","Z"]`, with
`"Z"` being the default basis where the quantum state is written.
If `numbases` is provided, the output consist of `numbases`
different measurement basis, each being repeated `nshots÷numbases`
times.
"""

function measurementsettings(N::Int,numshots::Int;
                             bases_id::Array=["X","Y","Z"],
                             numbases=nothing)
  # One shot per basis
  if isnothing(numbases)
    measurementbases = rand(bases_id,numshots,N)
  # Some number of shots per basis
  else
    @assert(numshots%numbases ==0)
    shotsperbasis = numshots÷numbases
    measurementbases = repeat(rand(bases_id,1,N),shotsperbasis)
    for n in 1:numbases-1
      newbases = repeat(rand(bases_id,1,N),shotsperbasis)
      measurementbases = vcat(measurementbases,newbases)
    end
  end
  return measurementbases
end


"""
  measurementgates(basis::Array)

Given as input a measurement basis, returns the corresponding
gate data structure. If the basis is `"Z"`, no action is required.
If not, a quantum gate corresponding to the given basis rotation
is added to the list.

Example:
  basis = ["X","Z","Z","Y"]
  -> gate_list = [("measX", 1),
                  ("measY", 4)]
"""
function measurementgates(basis::Array)
  gate_list = Tuple[]
  for j in 1:length(basis)
    if (basis[j]!= "Z")
      push!(gate_list,("meas$(basis[j])", j))
    end
  end
  return gate_list
end



"""
MEASUREMENT IN MULTIPLE BASES
"""
"""
  generatedata(M0::Union{MPS,MPO},basis::Array)

Generate a single measurements data-point in an input `basis`. 
If `Û` is the depth-1 local circuit rotating each qubit, the 
data-point `σ = (σ₁,σ₂,…)` id drawn from the probablity distribution:
- P(σ) = |⟨σ|Û|ψ⟩|²   : if M = ψ is MPS
- P(σ) = <σ|Û ρ Û†|σ⟩ : if M = ρ is MPO   
"""

function generatedata(M0::Union{MPS,MPO},basis::Array)
  # Generate basis rotation gates
  meas_gates = measurementgates(basis)
  # Apply basis rotation
  M = runcircuit(M0,meas_gates)
  # Measure
  measurement = generatedata(M,1)
  return convertdatapoint(measurement,basis)
end

"""
  generatedata(M::Union{MPS,MPO},nshots::Int,bases::Array)
Generate a dataset of `nshots` measurements acccording to a set
of input `bases`.
"""
function generatedata(M::Union{MPS,MPO},nshots::Int,bases::Array)
  data = Matrix{String}(undef, nshots,length(M))
  for n in 1:nshots
    data[n,:] = generatedata(M,bases[n,:])
  end
  return data 
end


""" 
  function generatedata(N::Int64,gates::Vector{<:Tuple},nshots::Int64;
                        noise=nothing,bases_id=nothing,return_state::Bool=false,
                        cutoff::Float64=1e-15,maxdim::Int64=10000,
                        kwargs...)

Generate `nshots` measurements at the output of a `N`-qubit quantum circuit, 
specified by a set of quantum `gates`.

Arguments:
  - `gates`: a set of quantum gates.
  - `noise`: apply a noise model after each quantum gate in the circuit
  - basis_id: set of basis used (ex: `basis_id=["X","Y","Z"])
  - return_state: if true, returns the ouput state `ψ = U|0,0,…,0⟩`
"""

function generatedata(N::Int64,gates::Vector{<:Tuple},nshots::Int64;
                      noise=nothing,bases_id=nothing,return_state::Bool=false,
                      cutoff::Float64=1e-15,maxdim::Int64=10000,
                      kwargs...)
  # Apply the quantum channel
  M = runcircuit(N,gates;process=false,noise=noise,
                 cutoff=cutoff,maxdim=maxdim,kwargs...)
  # Reference basis
  if isnothing(bases_id)
    data = generatedata(M,nshots)
  # Pauli bases
  else
    bases = measurementsettings(N,nshots;bases_id=bases_id)
    data = generatedata(M,nshots,bases)
  end
  return (return_state ? (M,data) : data)
end



"""
QUANTUM PROCESS TOMOGRAPHY
"""


"""
  preparationsettings(N::Int,nshots::Int;
                      prep_id::Array=["X+","X-","Y+","Y-","Z+","Z-"],
                      numprep=nothing)

Generate `nshots` input states to a quantum circuit. By default, each
single-qubit state is randomly selected between the 6 eigenstates of
the Pauli matrices, `["X+","X-","Y+","Y-","Z+","Z-"]`.
If `numprep` is provided, the output consist of `numprep`
different input states, each being repeated `nshots÷numprep`
times.
"""

function preparationsettings(N::Int,nshots::Int;
                             prep_id::Array=["X+","X-","Y+","Y-","Z+","Z-"],
                             numprep=nothing)
  # One shot per basis
  if isnothing(numprep)
    preparationstates = rand(prep_id,nshots,N)
  else
    @assert(nshots%numprep ==0)
    shotsperstate = nshots÷numprep
    preparationstates = repeat(rand(prep_id,1,N),shotsperstate)
    for n in 1:numprep-1
      newstates = repeat(rand(prep_id,1,N),shotsperstate)
      preparationstates = vcat(preparationstates,newstates)
    end
  end
  return preparationstates
end

"""
  measurementgates(basis::Array)

Given as input a prepared input state, returns the corresponding
gate data structure. If the state is `"Z+"`, no action is required.
If not, a quantum gate for state preparation is added to the list.

Example:
prep = ["X+","Z+","Z+","Y+"]
-> gate_list = [("prepX+", 1),
                ("prepY+", 4)]
"""

function preparationgates(prep::Array)
  gate_list = Tuple[]
  for j in 1:length(prep)
    if (prep[j]!= "Z+")
      gatename = "prep$(prep[j])"
      push!(gate_list, (gatename, j))
    end
  end
  return gate_list
end




""" 
  generate_processdata(M0::Union{MPS,MPO},
                       gate_tensors::Vector{<:ITensor},
                       prep::Array,basis::Array;
                       cutoff::Float64=1e-15,maxdim::Int64=10000,
                       kwargs...)


Generate a single data-point for quantum process tomography, 
consisting of an input state (a product state of single-qubit
Pauli eigenstates) and an output state measured after a given
basis rotation is performed at the output of a quantum channel.

Arguments:
 - `M0`: reference input state (to avoid re-compiling the circuit)
 - `gate_tensors`: tensors for the channel (unitary or noisy)
 - `prep`: a prepared input state (e.g. `["X+","Z-","Y+","X-"]`)
 - `basis`: a measuremement basis (e.g. `["Z","Z","Y","X"])
"""
# Generate a single data point for process tomography
function generate_processdata(M0::Union{MPS,MPO},
                              gate_tensors::Vector{<:ITensor},
                              prep::Array,basis::Array;
                              cutoff::Float64=1e-15,maxdim::Int64=10000,
                              kwargs...)
  # Generate preparation/measurement gates
  prep_gates = preparationgates(prep)
  meas_gates = measurementgates(basis)
  # Prepare quantum state
  M_in  = runcircuit(M0,prep_gates)
  # Apply the quantum channel
  M_out = runcircuit(M_in,gate_tensors,
                     cutoff=cutoff,maxdim=maxdim) 
  # Apply basis rotation
  M_meas = runcircuit(M_out,meas_gates)
  # Measure
  measurement = generatedata(M_meas,1)
  
  return convertdatapoint(measurement,basis)
end

"""
  projectchoi(Λ0::Union{MPS,MPO},prep::Array)

Project the Choi matrix (MPS/MPO) input indices into a state `prep` 
made out of single-qubit Pauli eigenstates (e.g. `|ϕ⟩ =|+⟩⊗|0⟩⊗|r⟩⊗…).
The resulting MPS/MPO describes the quantum state obtained by applying
the quantum channel underlying the Choi matrix to `|ϕ⟩`.

"""
function projectchoi(Λ0::Union{MPS,MPO},prep::Array)
  Λ = copy(Λ0) 
  state = "state" .* copy(prep) 
  
  M = ITensor[]
  s = (typeof(Λ)==MPS ? siteinds(Λ) : firstsiteinds(Λ))
  
  for j in 1:2:length(Λ)
    # No conjugate on the gate (transpose input)
    if typeof(Λ) == MPS
      Λ[j] = Λ[j] * gate(state[(j+1)÷2],s[j])
    else
      Λ[j] = Λ[j] * dag(gate(state[(j+1)÷2],s[j]))
      Λ[j] = Λ[j] * prime(gate(state[(j+1)÷2],s[j]))
    end
    push!(M,Λ[j]*Λ[j+1])
  end
  return (typeof(Λ) == MPS ? MPS(M) : MPO(M))
end

"""
  generate_processdata(Λ0::Union{MPS,MPO},prep::Array,basis::Array)

Generate a single data-point for quantum process tomography using an
input Choi matrix `Λ0`. Each data-point consists of an input state 
(a product state of single-qubit Pauli eigenstates) and an output 
state measured after a given basis rotation is performed at the output 
of a quantum channel.

"""

function generate_processdata(Λ0::Union{MPS,MPO},prep::Array,basis::Array)
  
  # Generate measurement gates
  meas_gates = measurementgates(basis)
  # Project Choi matrix input subspace
  Φ = projectchoi(Λ0,prep)
  # Apply basis rotation
  Φ_meas = runcircuit(Φ,meas_gates)
  # Measure
  measurement = generatedata(Φ_meas,1)
  return convertdatapoint(measurement,basis)
end

"""

generate_processdata(N::Int64,gates::Vector{<:Tuple},nshots::Int64;
             noise=nothing,choi::Bool=false,return_state::Bool=false,
             bases_id::Array=["X","Y","Z"],
             prep_id::Array=["X+","X-","Y+","Y-","Z+","Z-"],
             numbases=nothing,numprep=nothing,
             cutoff::Float64=1e-15,maxdim::Int64=10000,
             kwargs...)
Generate `nshots` data-point for quantum process tomography for a 
quantum channel corresponding to a set of quantum `gates` and a `noise`
model. 

Arguments:
  - `gates`: a set of quantum gates
  - `noise`: apply a noise model after each quantum gate in the circuit
  - `prep_id`: a set of input states (e.g. `["X+","X-","Y+","Y-","Z+","Z-"]`)   
  - `basis_id`: set of basis used (e.g. `["X","Y","Z"])
  - `choi`: if true, generate data using the Choi matrix
  - `return_state`: if true, returns the ouput state `ψ = U|0,0,…,0⟩`
"""

function generate_processdata(N::Int64,gates::Vector{<:Tuple},nshots::Int64;
                      noise=nothing,choi::Bool=false,return_state::Bool=false,
                      bases_id::Array=["X","Y","Z"],
                      prep_id::Array=["X+","X-","Y+","Y-","Z+","Z-"],
                      numbases=nothing,numprep=nothing,
                      cutoff::Float64=1e-15,maxdim::Int64=10000,
                      kwargs...)
  
  # Generate a set of prepared input state to the channel
  preps = preparationsettings(N,nshots,prep_id=prep_id,
                              numprep=numprep)
  # Generate a set of measurement bases at the output of the channel
  bases = measurementsettings(N,nshots,bases_id=bases_id,
                              numbases=numbases)
  
  # Generate data with Choi matrix
  if choi
    # Compute Choi matrix
    Λ = choimatrix(N,gates;noise=noise,cutoff=cutoff,maxdim=maxdim,kwargs...)
    # Generate data
    data = Matrix{String}(undef, nshots,length(Λ)÷2)
    for n in 1:nshots
      data[n,:] = generate_processdata(Λ,preps[n,:],bases[n,:])
    end
    
    return (return_state ? (Λ ,preps, data) : (preps,data))
  else
    # Initialize state and indiccecs
    ψ0 = qubits(N)
    # Pre-compile quantum channel
    gate_tensors = compilecircuit(ψ0,gates; noise=noise, kwargs...)
    # Generate data
    data = Matrix{String}(undef, nshots,length(ψ0))
    for n in 1:nshots
      data[n,:] = generate_processdata(ψ0,gate_tensors,preps[n,:],bases[n,:];
                              noise=noise,cutoff=cutoff,choi=false,
                              maxdim=maxdim,kwargs...)
    end
    return (preps,data) 
  end
end


"""
  convertdatapoint(datapoint::Array,basis::Array;state::Bool=false)

Convert a data point from (sample,basis) -> data
Ex: (0,1,0,0) (X,Z,Y,X) -> (X+,Z-,Y+,X+)
"""

function convertdatapoint(datapoint::Array,basis::Array;state::Bool=false)
  newdata = []
  for j in 1:length(datapoint)
    if basis[j] == "X"
      if datapoint[j] == 0
        dat = (state ? "stateX+" : "X+")
        push!(newdata,dat)
      else
        dat = (state ? "stateX-" : "X-")
        push!(newdata,dat)
      end
    elseif basis[j] == "Y"
      if datapoint[j] == 0
        dat = (state ? "stateY+" : "Y+")
        push!(newdata,dat)
      else
        dat = (state ? "stateY-" : "Y-")
        push!(newdata,dat)
      end
    elseif basis[j] == "Z"
      if datapoint[j] == 0
        dat = (state ? "stateZ+" : "Z+")
        push!(newdata,dat)
      else
        dat = (state ? "stateZ-" : "Z-")
        push!(newdata,dat)
      end
    end
  end
  return newdata
end

function convertdatapoints(datapoints::Array,bases::Array;state::Bool=false)
  newdata = Matrix{String}(undef, size(datapoints)[1],size(datapoints)[2]) 
  
  for n in 1:size(datapoints)[1]
    newdata[n,:] = convertdatapoint(datapoints[n,:],bases[n,:],state=state)
  end
  return newdata
end


