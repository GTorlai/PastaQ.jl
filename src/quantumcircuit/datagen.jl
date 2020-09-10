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
DATA GENERATION AT CIRCUIT OUTPUT
"""

"""
Perform a projective measurements on a wavefunction or density operator
"""
function measure(M::Union{MPS,MPO},nshots::Int)
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

function generatedata(N::Int64,gates::Vector{<:Tuple},nshots::Int64; 
                      noise=nothing,process=false,
                      cutoff::Float64=1e-15,maxdim::Int64=10000,
                      bases_id::Array=["X","Y","Z"],
                      prep_id::Array=["X+","X-","Y+","Y-","Z+","Z-"],
                      numbases=nothing,numprep=nothing,
                      kwargs...)
  
  # Measure at the output of the circuit
  if process==false
    M = runcircuit(N,gates;process=false,noise=noise,
                   cutoff=cutoff,maxdim=maxdim,
                   kwargs...)
    bases = measurementsettings(N,nshots,bases_id=bases_id,
                                numbases=numbases)
    generatedata(M,nshots,bases) 
  
  # Generate data for process tomography
  else
    preps = preparationsettings(N,nshots,prep_id=prep_id,
                                numprep=numprep)
    
    bases = measurementsettings(N,nshots,bases_id=bases_id,
                                numbases=numbases)
    ψ0 = qubits(N)
    gate_tensors = compilecircuit(ψ0,gates; noise=noise, kwargs...)
    data_in  = preps
    data_out = generatedata(ψ0,gate_tensors,nshots,preps,bases;
                            noise=noise,cutoff=cutoff,
                            maxdim=maxdim,kwargs...) 
    return (data_in,data_out)
  end
end

"""
STATE PREPARATION + MEASUREMENT AT THE OUTPUT
"""

# Generate a single data point for process tomography
function generatedata(M0::Union{MPS,MPO},
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
  measurement = measure(M_meas,1)
  return convertdatapoint(measurement,basis)
end

# Generate nshots data points for process tomography
function generatedata(M0::Union{MPS,MPO},
                      gate_tensors::Vector{<:ITensor},
                      nshots::Int64,preps::Array,bases::Array;
                      noise=nothing,
                      cutoff::Float64=1e-15,maxdim::Int64=10000,
                      kwargs...)
  data = Matrix{String}(undef, nshots,length(M0))
  for n in 1:nshots
    data[n,:] = generatedata(M0,gate_tensors,preps[n,:],bases[n,:];
                            noise=noise,cutoff=cutoff,
                            maxdim=maxdim,kwargs...)
  end
  return data  
end

"""
MEASUREMENT AT THE OUTPUT
"""

# Generate a single data point in a basis for MPS/MPO
function generatedata(M0::Union{MPS,MPO},basis::Array)
  meas_gates = measurementgates(basis)
  M = runcircuit(M0,meas_gates)
  measurement = measure(M,1)
  return convertdatapoint(measurement,basis)
end

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

#function convertdatapoints(datapoints::Array,bases::Array;state::Bool=false)
#  newdata = Matrix{String}(undef, size(datapoints)[1],size(datapoints)[2]) 
#  
#  for n in 1:size(datapoints)[1]
#    newdata[n,:] = convertdatapoint(datapoints[n,:],bases[n,:],state=state)
#  end
#  return newdata
#end
#

