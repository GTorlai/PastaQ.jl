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
                      model::Union{MPS, MPO, LPDO, Nothing},
                      output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "outcomes", data)
    if isnothing(model)
      write(fout, "model", "nothing")
    else
      write(fout, "model", model)
    end
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
                      model::Union{MPS, MPO, LPDO, Nothing},
                      output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "bases", first.(data))
    write(fout, "outcomes", last.(data))
    if isnothing(model)
      write(fout, "model", "nothing")
    else
      write(fout, "model", model)
    end
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
                      model::Union{MPS, MPO, LPDO, Nothing},
                      output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "inputs", first.(data))
    write(fout, "bases", first.(last.(data)))
    write(fout, "outcomes", last.(last.(data)))
    if isnothing(model)
      write(fout, "model", "nothing")
    else
      write(fout, "model", model)
    end
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
    data = read(fin, "outcomes")
  else
    close(fin)
    error("File must contain either \"data\" for quantum state tomography data or \"data_first\" and \"data_second\" for quantum process tomography.")
  end

  # Check if a model is saved, if so read it and return it
  if exists(fin, "model")
    g = fin["model"]

    if exists(attrs(g), "type")
      typestring = read(attrs(g)["type"])
      modeltype = eval(Meta.parse(typestring))
      model = read(fin, "model", modeltype)
    else
      model = read(fin, "model")
      if model == "nothing"
        model = nothing
      else
        error("model must be MPS, LPDO, Choi, or Nothing")
      end
    end
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

function convertdatapoints(datapoints::Array{String})
  nshots = size(datapoints)[1]
  newdata = Matrix{Pair{String,Int64}}(undef,nshots,size(datapoints)[2])

  for n in 1:nshots
    newdata[n,:] = convertdatapoint(datapoints[n,:])
  end
  return newdata
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

"""
    split_dataset(data::Matrix; train_ratio::Float64 = 0.9, randomize::Bool = true)

Split a data set into a `train` and `test` sets, given a `train_ratio` (i.e. the 
percentage of data in `train_data`. If `randomize=true` (default), the `data` is 
randomly shuffled before splitting.
"""
function split_dataset(data::Matrix; train_ratio::Float64 = 0.9, randomize::Bool = true)
  ndata = size(data,1)
  ntrain = Int(ndata * train_ratio)
  ntest = ndata - ntrain
  if randomize
    data = data[shuffle(1:end),:]
  end
  train_data = data[1:ntrain,:]
  test_data  = data[ntrain+1:end,:]
  return train_data,test_data
end



function ischoi(M::LPDO)
  return (length(inds(M.X[1],"Site")) == 2 ? true : false)
end

function ischoi(M::MPO)
  return (length(inds(M[1],"Site")) == 4 ? true : false)
  #return ( length(inds(M[1])) == 5 ? true : false)
end

function makeUnitary(L::LPDO{MPS})
  ψ = L.X
  U = MPO(ITensor[copy(ψ[j]) for j in 1:length(ψ)])
  prime!(U,tags="Output")
  removetags!(U, "Input")
  removetags!(U, "Output")
  return U
end

function makeChoi(U0::MPO)
  M = MPS(ITensor[copy(U0[j]) for j in 1:length(U0)])
  addtags!(M, "Input", plev = 0, tags = "Qubit")
  addtags!(M, "Output", plev = 1, tags = "Qubit")
  noprime!(M)
  return LPDO(M)
end


function numberofqubits(gate::Tuple)
  s = gate[2]
  n = (s isa Number ? s : maximum(s))
  return n
end

function numberofqubits(gates::Vector{<:Tuple})
  nMax = 0
  for g in gates
    s = g[2]
    n = (s isa Number ? s : maximum(s))
    nMax = (n > nMax ? n : nMax)
  end
  return nMax
end



