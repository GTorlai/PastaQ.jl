"""
    array(M::MPS; reverse::Bool = true)

Generate the full dense vector from an MPS
"""
function array(M::MPS; reverse::Bool = true)
  s = siteinds(M)
  if reverse
    s = Base.reverse(s)
  end
  C = combiner(s...)
  Mvec = prod(M) * dag(C)
  return array(Mvec)
end

"""
    array(M::MPO; reverse::Bool = true)
    array(L::LPDO; reverse::Bool = true)

Generate the full dense matrix from an MPO or LPDO.
"""
function array(M::MPO; reverse::Bool = true)
  s = firstsiteinds(M; plev = 0)
  if reverse
    s = Base.reverse(s)
  end
  C = combiner(s...)
  Mmat = prod(M) * dag(C) * C'
  c = combinedind(C)
  return array(permute(Mmat, c', c))
end

function array(L::LPDO{MPO}; kwargs...) 
  !ischoi(L) && return array(MPO(L); kwargs...)
  error("array function for Choi matrix LPDO not yet implemented")
end

# TEMPORARY FUNCTION
# TODO: remove when `firstsiteinds(ψ::MPS)` is implemented
function hilbertspace(L::LPDO)
  return  (L.X isa MPS ? siteinds(L.X) : firstsiteinds(L.X))
end

hilbertspace(M::Union{MPS,MPO}) = hilbertspace(LPDO(M))

#function replace_hilbertspace!(M::Union{MPS,MPO}, REF::Union{MPS,MPO,LPDO})
#  make_inds_match = true
#  siteindsM = siteinds(all, M)
#  siteindsREF = siteinds(all, REF)
#  if any(n -> length(n) > 1, siteindsM) ||
#     any(n -> length(n) > 1, siteindsREF) ||
#    !ITensors.hassamenuminds(siteinds, M, REF)
#    make_inds_match = false
#  end
#  if make_inds_match
#    ITensors.replace_siteinds!(M, siteindsREF)
#  end
#end






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
  ischoi(U0) && return U0
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

function numberofqubits(gates::Vector{<:Vector{<:Tuple}})
  nMax = 0
  for layer in gates
    for g in layer
      s = g[2]
      n = (s isa Number ? s : maximum(s))
      nMax = (n > nMax ? n : nMax)
    end
  end
  return nMax
end



