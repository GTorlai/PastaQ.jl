"""
    writesamples(data::Matrix,
                 [model::Union{MPS, MPO, LPDO, Choi},]
                 output_path::String)

Save data and model on file:

# Arguments:
  - `data`: array of measurement data
  - `model`: (optional) MPS, MPO, or Choi
  - `output_path`: path to file
"""
function writesamples(data::Matrix{Int},
                      model::Union{MPS, MPO, LPDO, Choi},
                      output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "outcomes", data)
    write(fout,"model", model)
  end
end

function writesamples(data::Matrix{Int},
                      output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "outcomes", data)
  end
end

function writesamples(data::Matrix{Pair{String, Int}},
                      model::Union{MPS, MPO, LPDO, Choi},
                      output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "bases", first.(data))
    write(fout, "outcomes", last.(data))
    write(fout,"model",model)
  end
end

function writesamples(data::Matrix{Pair{String, Int}},
                      output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "bases", first.(data))
    write(fout, "outcomes", last.(data))
  end
end

function writesamples(data::Matrix{Pair{String, Pair{String, Int}}},
                      model::Union{MPS, MPO, LPDO, Choi},
                      output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "inputs", first.(data))
    write(fout, "bases", first.(last.(data)))
    write(fout, "outcomes", last.(last.(data)))
    write(fout, "model", model)
  end
end

function writesamples(data::Matrix{Pair{String, Pair{String, Int}}},
                      output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "inputs", first.(data))
    write(fout, "bases", first.(last.(data)))
    write(fout, "outcomes", last.(last.(data)))
  end
end

"""
    readsamples(input_path::String)

Load data and model from file:

# Arguments:
  - `input_path`: path to file
"""
function readsamples(input_path::String)
  fin = h5open(input_path, "r")
  # Check if the data is for state tomography or process tomography
  # Process tomography
  if exists(fin, "inputs")
    inputs = read(fin, "inputs")
    bases = read(fin, "bases")
    outcomes = read(fin,"outcomes")
    data = inputs .=> (bases .=> outcomes)
  # Measurements in bases
  elseif exists(fin, "bases") 
    bases = read(fin, "bases")
    outcomes = read(fin,"outcomes")
    data = bases .=> outcomes
  # Measurements in Z basis
  elseif exists(fin, "outcomes")
    data = read(fin,"outcomes")
  else
    close(fin)
    error("File must contain either \"data\" for quantum state tomography data or \"data_first\" and \"data_second\" for quantum process tomography.")
  end

  # Check if a model is saved, if so read it and return it
  if exists(fin, "model")
    g = g_open(fin, "model")
    typestring = read(attrs(g)["type"])
    modeltype = eval(Meta.parse(typestring))
    model = read(fin, "model", modeltype)
    close(fin)
    return data, model
  end

  close(fin)
  return data
end

"""
    PastaQ.fullvector(M::MPS; reverse::Bool = true)

Extract the full vector from an MPS
"""
function fullvector(M::MPS; reverse::Bool = true)
  s = siteinds(M)
  if reverse
    s = Base.reverse(s)
  end
  C = combiner(s...)
  Mvec = prod(M) * dag(C)
  return array(Mvec)
end

"""
    PastaQ.fullmatrix(M::MPO; reverse::Bool = true)
    PastaQ.fullmatrix(L::LPDO; reverse::Bool = true)

Extract the full matrix from an MPO or LPDO, returning a Julia Matrix.
"""
function fullmatrix(M::MPO; reverse::Bool = true)
  s = firstsiteinds(M; plev = 0)
  if reverse
    s = Base.reverse(s)
  end
  C = combiner(s...)
  Mmat = prod(M) * dag(C) * C'
  c = combinedind(C)
  return array(permute(Mmat, c', c))
end

fullmatrix(L::LPDO; kwargs...) = fullmatrix(MPO(L); kwargs...)

# TEMPORARY FUNCTION
# TODO: remove when `firstsiteinds(ψ::MPS)` is implemented
function hilbertspace(L::LPDO)
  return  (L.X isa MPS ? siteinds(L.X) : firstsiteinds(L.X))
end

hilbertspace(M::Union{MPS,MPO}) = hilbertspace(LPDO(M))




"""
    convertdatapoint(datapoint::Array,basis::Array;state::Bool=false)

0 1 0 0 1 -> Z+ Z- Z+ Z+ Z-
"""
function convertdatapoint(datapoint::Array{Int64};
                          state::Bool=false)
  newdata = []
  basis = ["Z" for _ in 1:length(datapoint)]
  for j in 1:length(datapoint)
    if datapoint[j] == 0
      push!(newdata,"Z+")
    elseif datapoint[j] == 1
      push!(newdata,"Z-")
    else
      error("non-binary data")
    end
  end
  return newdata
end

"""

(Z+, X-) -> (Z => 0), (X => 1)   

"""
function convertdatapoint(datapoint::Array{String})
  basis = []
  outcome = []
  for j in 1:length(datapoint)
    push!(basis,string(datapoint[j][1]))
    
    if datapoint[j][2] == Char('+') 
      push!(outcome,0)
    elseif datapoint[j][2] == Char('-')
      push!(outcome,1)
    else
      error("non-binary data")
    end
  end
  return basis .=> outcome
end

"""

(Z, 0), (X, 1)  / (Z+, X-)

"""
function convertdatapoint(outcome::Array{Int64}, basis::Array{String};
                          state::Bool=false)
  @assert length(outcome) == length(basis)
  newdata = []
  if state
    basis = basis
  end
  for j in 1:length(outcome)
    if outcome[j] == 0 
      push!(newdata, basis[j] * "+")
    elseif outcome[j] == 1
      push!(newdata, basis[j] * "-")
    else
      error("non-binary data")
    end
  end
  return newdata
end

convertdatapoint(datapoint::Array{Pair{String,Int64}}; state::Bool=false) = 
  convertdatapoint(last.(datapoint),first.(datapoint); state = state)


function convertdatapoints(datapoints::Array{Pair{String,Int64}}; state::Bool=false)
  nshots = size(datapoints)[1]
  newdata = Matrix{String}(undef,nshots,size(datapoints)[2])

  for n in 1:nshots
    newdata[n,:] = convertdatapoint(datapoints[n,:]; state = state)
  end
  return newdata
end

function convertdatapoints(outcome::Matrix{Int64}, basis::Matrix{String}; state::Bool=false)
  nshots = size(datapoints)[1]
  newdata = Matrix{String}(undef,nshots,size(datapoints)[2])

  for n in 1:nshots
    newdata[n,:] = convertdatapoint(datapoints[n,:]; state = state)
  end
  return newdata
end
"""

(Z => 0), (X => 1)  / (Z+, X-)

"""






#function convertdatapoint(datapoint::Array{Pair{String,Int64}})
#  newdata = []
#  basis = first.(datapoint)
#  outcome = last.(datapoint)
#
#  for j in 1:length(datapoint)
#    if outcome[j] == 0 
#      push!(newdata, basis[j] * "+")
#    elseif outcome[j] == 1
#      push!(newdata, basis[j] * "-")
#    else
#      error("non-binary data")
#    end
#  end
#  return newdata
#end

#function convertdatapoints(datapoints::Matrix{Pair{String,Int64}})
#  nshots = size(datapoints)[1]
#  newdata = Matrix{String}(undef,nshots,size(datapoints)[2])
#
#  for n in 1:nshots
#    newdata[n,:] = convertdatapoint(datapoints[n,:])
#  end
#  return newdata
#end



#function convertdatapoint(datapoint::Array{Int64}, basis::Array{String})
#  newdata = []
#  for j in 1:length(datapoint)
#    push!(newdata,basis[j] => datapoint[j])
#  end
#  return newdata
#end
#
#"""
#    convertdatapoint(datapoint::Array,basis::Array;state::Bool=false)
#
#("Z" => 0, "X" => 1) -> ("Z+","X-") 
#"""

#"""
#    convertdatapoint(datapoint::Array,basis::Array;state::Bool=false)
#
#Many points: ("Z" => 0, "X" => 1) -> ("Z+","X-") 
#"""
#function convertdatapoints(datapoints::Matrix{Pair{String,Int64}}; state::Bool=false)
#  nshots = size(datapoints)[1]
#  newdata = Matrix{String}(undef,nshots,size(datapoints)[2])
#
#  for n in 1:nshots
#    newdata[n,:] = convertdatapoint(datapoints[n,:]; state = state)
#  end
#  return newdata
#end
#
##function convertdatapoints(datapoints::Array,
##                           bases::Array;
##                           state::Bool=false)
##  newdata = Matrix{String}(undef, size(datapoints)[1],size(datapoints)[2]) 
##  
##  for n in 1:size(datapoints)[1]
##    newdata[n,:] = convertdatapoint(datapoints[n,:],bases[n,:],state=state)
##  end
##  return newdata
##end
#
