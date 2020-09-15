"""
Perform a projective measurements on a wavefunction or density operator
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
Given as input a measurement basis, returns the corresponding
gate data structure.
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
Generate a set of measurement bases:
- nshots = total number of bases
if numbases=nothing: nshots different bases
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
MEASUREMENT IN MULTIPLE BASES
"""

# Generate nshots datapoints in bases for MPS/MPO
"""
Generate a dataset of measurements in different bases
"""
function generatedata(M::Union{MPS,MPO},nshots::Int,bases::Array)
  data = Matrix{String}(undef, nshots,length(M))
  for n in 1:nshots
    data[n,:] = generatedata(M,bases[n,:])
  end
  return data 
end

# Generate a single data point in a basis for MPS/MPO
function generatedata(M0::Union{MPS,MPO},basis::Array)
  meas_gates = measurementgates(basis)
  M = runcircuit(M0,meas_gates)
  measurement = generatedata(M,1)
  return convertdatapoint(measurement,basis)
end

""" 
Generate data at the output of a circuit
"""

function generatedata(N::Int64,gates::Vector{<:Tuple},nshots::Int64;
                      noise=nothing,bases_id=nothing,return_state::Bool=false,
                      cutoff::Float64=1e-15,maxdim::Int64=10000,
                      kwargs...)
  M = runcircuit(N,gates;process=false,noise=noise,
                 cutoff=cutoff,maxdim=maxdim,kwargs...)
  if isnothing(bases_id)
    data = generatedata(M,nshots)
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
Given as input a preparation state, returns the corresponding
gate data structure.
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
Generate a set of preparation states:
- nshots = total number of states
if numprep=nothing: nshots different states
"""
function preparationsettings(N::Int,numshots::Int;
                             prep_id::Array=["X+","X-","Y+","Y-","Z+","Z-"],
                             numprep=nothing)
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
Generate data at the output of a circuit
"""
# Generate a single data point for process tomography
function generate_processdata(M0::Union{MPS,MPO},
                              gate_tensors::Vector{<:ITensor},
                              prep::Array,basis::Array;
                              noise=nothing,
                              cutoff::Float64=1e-15,maxdim::Int64=10000,
                              kwargs...)
  prep_gates = preparationgates(prep)
  meas_gates = measurementgates(basis)
  M_in  = runcircuit(M0,prep_gates)
  M_out = runcircuit(M_in,gate_tensors,
                     cutoff=cutoff,maxdim=maxdim) 
  M_meas = runcircuit(M_out,meas_gates)
  measurement = generatedata(M_meas,1)
  
  return convertdatapoint(measurement,basis)
end

#TODO
# Fix inconstincenice with prroject choi
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

# Generate a single data point for process tomography
function generate_processdata(Λ0::Union{MPS,MPO},prep::Array,basis::Array)
  
  meas_gates = measurementgates(basis)
  ϕ = projectchoi(Λ0,prep)
  ϕ_meas = runcircuit(ϕ,meas_gates)
  measurement = generatedata(ϕ_meas,1)
  return convertdatapoint(measurement,basis)
end


function generate_processdata(N::Int64,gates::Vector{<:Tuple},nshots::Int64;
                      noise=nothing,choi::Bool=false,return_state::Bool=false,
                      bases_id::Array=["X","Y","Z"],
                      prep_id::Array=["X+","X-","Y+","Y-","Z+","Z-"],
                      numbases=nothing,numprep=nothing,
                      cutoff::Float64=1e-15,maxdim::Int64=10000,
                      kwargs...)
   
    preps = preparationsettings(N,nshots,prep_id=prep_id,
                                numprep=numprep)
    
    bases = measurementsettings(N,nshots,bases_id=bases_id,
                                numbases=numbases)

    if choi
      Λ = choimatrix(N,gates;noise=noise,cutoff=cutoff,maxdim=maxdim,kwargs...)
      data = Matrix{String}(undef, nshots,length(Λ)÷2)
      
      for n in 1:nshots
        data[n,:] = generate_processdata(Λ,preps[n,:],bases[n,:])
      end
      
      return (return_state ? (Λ ,preps, data) : (preps,data))
    else
      ψ0 = qubits(N)
      gate_tensors = compilecircuit(ψ0,gates; noise=noise, kwargs...)
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


