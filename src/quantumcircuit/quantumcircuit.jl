"""----------------------------------------------
                  INITIALIZATION 
------------------------------------------------- """

"""
Initialize MPS wavefunction |ψ⟩
"""
wavefunction(sites::Vector{<:Index}) = productMPS(sites, "0")

wavefunction(N::Int) = wavefunction(siteinds("qubit", N))

""" 
Initialize MPO density matrix ρ
"""
densitymatrix(sites::Vector{<:Index}) = 
  densitymatrix(productMPS(sites, "0"))

densitymatrix(N::Int) = 
  densitymatrix(siteinds("qubit",N))

""" 
Initialize qubits
"""
qubits(sites::Vector{<:Index}; mixed::Bool=false) = 
  mixed ? densitymatrix(sites) : wavefunction(sites) 

qubits(N::Int; mixed::Bool=false) = qubits(siteinds("qubit", N); mixed=mixed)

""" 
Reset qubits to the initial state
"""
function resetqubits!(M::Union{MPS,MPO})
  indices = [firstind(M[j],tags="Site",plev=0) for j in 1:length(M)]
  M_new = (typeof(M) == MPS ? wavefunction(indices) : densitymatrix(indices))
  M[:] = M_new
  return M
end

"""
Build a density matrix ρ = |ψ⟩⟨ψ|
"""
function densitymatrix(ψ::MPS)
  ρ = MPO([ψ[n]' * dag(ψ[n]) for n in 1:length(ψ)])
  for n in 1:length(ρ)-1 
    C = combiner(commoninds(ρ[n], ρ[n+1]))
    ρ[n] *= C
    ρ[n+1] *= dag(C)
  end
  return ρ
end

"""
Initialize a circuit MPO
"""
circuit(sites::Vector{<:Index}) = MPO(sites, "Id")

circuit(N::Int) = circuit(siteinds("qubit", N))

"""
Initialize the Choi matrix Λ
"""
choi(sites::Vector{<:Index}; mixed::Bool=false) = 
  qubits(sites; mixed = mixed) 

choi(N::Int; mixed::Bool=false) = 
  choi(siteinds("qubit",2*N); mixed = mixed) 


"""----------------------------------------------
                  CIRCUIT FUNCTIONS 
------------------------------------------------- """

"""
Add a list of gates to gates (data structure) 
"""
function appendgates!(gates::Vector{<:Tuple},newgates::Vector{<:Tuple})
  for newgate in newgates
    push!(gates,newgate)
  end
end

"""
Generates a vector of ITensors from a tuple of gates
"""
function compilecircuit(M::Union{MPS,MPO},gates::Vector{<:Tuple}; 
                        noise=nothing, kwargs...)
  gate_tensors = ITensor[]
  for g in gates
    push!(gate_tensors,makegate(M,g))
    if !isnothing(noise)
      if (typeof(g[2]) == Int)
        push!(gate_tensors,makekraus(M,noise,g[2];kwargs...))
      else
        gate_tensors = cat(gate_tensors,makekraus(M,noise,g[2];kwargs...),dims=1)
      end
    end
  end
  return gate_tensors
end

"""
Apply the circuit to a state M using a set of gate tensors
"""
function runcircuit(M::Union{MPS,MPO},gate_tensors::Vector{<:ITensor}; kwargs...) 
  # Check if gate_tensors contains Kraus operators
  inds_sizes = [length(inds(g)) for g in gate_tensors]
  noiseflag = any(x -> x%2==1 , inds_sizes)
  
  buildMPO::Bool = get(kwargs,:get_unitary,false)
  
  if buildMPO & !noiseflag
    Mc = apply(reverse(gate_tensors),M; kwargs...) 
  elseif buildMPO & noiseflag
    error("Cannot construct a unitary MPO for a noisy circuit")
  elseif noiseflag
    ρ = (typeof(M) == MPS ? densitymatrix(M) : M)
    Mc = apply(reverse(gate_tensors),ρ; apply_dag=true, kwargs...)
  else
    Mc = (typeof(M) == MPS ? apply(reverse(gate_tensors),M; kwargs...) :
                             apply(reverse(gate_tensors), M; apply_dag=true, kwargs...))
  end
  return Mc
end

"""
Apply the circuit on state M using a set of gates (Tuple)
"""
function runcircuit(M::Union{MPS,MPO},gates::Vector{<:Tuple}; noise=nothing, 
                    cutoff=1e-15,maxdim=10000,kwargs...)
  gate_tensors = compilecircuit(M,gates; noise=noise, kwargs...) 
  runcircuit(M,gate_tensors; kwargs...)
end

"""
Apply the circuit to a ITensor 
"""
function runcircuit(M::ITensor,gates::Vector{<:Tuple}; cutoff=1e-15,maxdim=10000,kwargs...)
  gate_tensors = compilecircuit(M,gates)
  return runcircuit(M,gate_tensors;cutoff=1e-15,maxdim=10000,kwargs...)
end

runcircuit(M::ITensor,
           gate_tensors::Vector{ <: ITensor};
           kwargs...) =
  apply(reverse(gate_tensors), M; kwargs...)


"""----------------------------------------------
               MEASUREMENT FUNCTIONS 
------------------------------------------------- """

"""
Given as input a measurement basis, returns the corresponding
gate data structure.
Example:
basis = ["X","Z","Z","Y"]
-> gate_list = [("measX", 1),
                ("measY", 4)]
"""
function makemeasurementgates(basis::Array)
  gate_list = Tuple[]
  for j in 1:length(basis)
    if (basis[j]!= "Z")
      push!(gate_list,("meas$(basis[j])", j))
    end
  end
  return gate_list
end

"""
Given as input a preparation state, returns the corresponding
gate data structure.
Example:
prep = ["X+","Z+","Z+","Y+"]
-> gate_list = [("prepX+", 1),
                ("prepY+", 4)]
"""
function makepreparationgates(prep::Array)
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
Generate a set of measurement bases:
- nshots = total number of bases
if numbases=nothing: nshots different bases
"""
function generatemeasurementsettings(N::Int,numshots::Int;
                                     numbases=nothing,bases_id=nothing)
  if isnothing(bases_id)
    bases_id = ["X","Y","Z"]
  end
  # One shot per basis
  if isnothing(numbases)
    measurementbases = rand(bases_id,numshots,N)
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
Generate a set of preparation states:
- nshots = total number of states
if numprep=nothing: nshots different states
"""
function generatepreparationsettings(N::Int,numshots::Int;numprep=nothing,prep_id=nothing)
  if isnothing(prep_id)
    prep_id = ["X+","X-","Y+","Y-","Z+","Z-"]
  end
  # One shot per basis
  if isnothing(numprep)
    preparationstates = rand(prep_id,numshots,N)
  else
    @assert(numshots%numprep ==0)
    shotsperstate = numshots÷numprep
    preparationstates = repeat(rand(prep_id,1,N),shotsperstate)
    for n in 1:numprep-1
      newstates = repeat(rand(prep_id,1,N),shotsperstate)
      preparationstates = vcat(preparationstates,newstates)
    end
  end
  return preparationstates
end

"""
Perform a projective measurements on a wavefunction
"""
function measure(mps::MPS,nshots::Int)
  orthogonalize!(mps,1)
  if (nshots==1)
    measurements = sample(mps)
    measurements .-= 1
  else
    measurements = Matrix{Int64}(undef, nshots, length(mps))
    for n in 1:nshots
      measurement = sample(mps)
      measurement .-= 1
      measurements[n,:] = measurement
    end
  end
  return measurements
end

"""
Generate a dataset of measurements in different bases
"""
function generatedata(psi::MPS,nshots::Int,bases::Array)
  data = Matrix{String}(undef, nshots,length(psi))
  for n in 1:nshots
    meas_gates = makemeasurementgates(bases[n,:])
    meas_tensors = compilecircuit(psi,meas_gates)
    psi_out = runcircuit(psi,meas_tensors)
    measurement = measure(psi_out,1)
    data[n,:] = convertdata(measurement,bases[n,:])
  end
  return data 
end

"""
Convert a data point from (sample,basis) -> data
Ex: (0,1,0,0) (X,Z,Y,X) -> (X+,Z-,Y+,X+)
"""
function convertdata(datapoint::Array,basis::Array)
  newdata = []
  for j in 1:length(datapoint)
    if basis[j] == "X"
      if datapoint[j] == 0
        push!(newdata,"projX+")
      else
        push!(newdata,"projX-")
      end
    elseif basis[j] == "Y"
      if datapoint[j] == 0
        push!(newdata,"projY+")
      else
        push!(newdata,"projY-")
      end
    elseif basis[j] == "Z"
      if datapoint[j] == 0
        push!(newdata,"projZ+")
      else
        push!(newdata,"projZ-")
      end
    end
  end
  return newdata
end

"""
Append a layer of Hadamard gates
"""
function hadamardlayer!(gates::Array,N::Int)
  for j in 1:N
    push!(gates,("H", j))
  end
end

"""
Append a layer of random single-qubit rotations
"""
function rand1Qrotationlayer!(gates::Array,N::Int;
                              rng=nothing)
  for j in 1:N
    if isnothing(rng)
      θ,ϕ,λ = rand!(zeros(3))
    else
      θ,ϕ,λ = rand!(rng,zeros(3))
    end
    push!(gates,("Rn", j, (θ = π*θ, ϕ = 2*π*ϕ, λ = 2*π*λ)))
  end
end

"""
Append a layer of CX gates
"""
function CXlayer!(gates::Array,N::Int;sequence::String)
  if (N ≤ 2)
    throw(ArgumentError("CXlayer is defined for N ≥ 3"))
  end
  
  if sequence == "odd"
    for j in 1:2:(N-N%2)
      push!(gates,("CX", (j,j+1)))
    end
  elseif sequence == "even"
    for j in 2:2:(N+N%2-1)
      push!(gates,("CX", (j,j+1)))
    end
  else
    throw(ArgumentError("Sequence not recognized"))
  end
end

"""
Generate a random quantum circuit
"""
function randomquantumcircuit(N::Int,depth::Int;rng=nothing)
  gates = Tuple[]
  for d in 1:depth
    rand1Qrotationlayer!(gates,N,rng=rng)
    if d%2==1
      CXlayer!(gates,N,sequence="odd")
    else
      CXlayer!(gates,N,sequence="even")
    end
  end
  return gates
end

