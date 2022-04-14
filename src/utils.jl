_convert_leaf_eltype(T::Type, x) = convert_leaf_eltype(T, x)
_convert_leaf_eltype(::Nothing, x) = x

convert_to_full_representation(M::ITensor) = M
convert_to_full_representation(M::MPS) = prod(M)
convert_to_full_representation(M::MPO) = prod(M)

# TODO: turn this into an ITensors.jl function `originalsiteinds`
# that generically returns the site indices that would be used to
# make an object of the same type with the same indices.
originalsiteinds(M::MPS)  = siteinds(first, M; plev=0)
originalsiteinds(M::MPO)  = dag.(siteinds(first, M; plev=0))
originalsiteinds(L::LPDO) = dag.(siteinds(first, L.X; plev=0, tags=!purifier_tags(L)))

function originalsiteinds(M::ITensor)
  s = inds(M; plev = 0)
  sp = sortperm(collect(string.(tags.(s))))
  return collect(s[sp])
end


"""
    convertdatapoint(datapoint::Array,basis::Array;state::Bool=false)

0 1 0 0 1 -> Z+ Z- Z+ Z+ Z-
"""
function convertdatapoint(datapoint::Array{Int64}; state::Bool=false)
  newdata = []
  basis = ["Z" for _ in 1:length(datapoint)]
  for j in 1:length(datapoint)
    if datapoint[j] == 0
      push!(newdata, "Z+")
    elseif datapoint[j] == 1
      push!(newdata, "Z-")
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
    push!(basis, string(datapoint[j][1]))

    if datapoint[j][2] == Char('+')
      push!(outcome, 0)
    elseif datapoint[j][2] == Char('-')
      push!(outcome, 1)
    else
      error("non-binary data")
    end
  end
  return basis .=> outcome
end

function convertdatapoints(datapoints::Array{String})
  nshots = size(datapoints)[1]
  newdata = Matrix{Pair{String,Int64}}(undef, nshots, size(datapoints)[2])

  for n in 1:nshots
    newdata[n, :] = convertdatapoint(datapoints[n, :])
  end
  return newdata
end

"""

(Z, 0), (X, 1)  / (Z+, X-)

"""
function convertdatapoint(outcome::Array{Int64}, basis::Array{String}; state::Bool=false)
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

function convertdatapoint(datapoint::Array{Pair{String,Int64}}; state::Bool=false)
  return convertdatapoint(last.(datapoint), first.(datapoint); state=state)
end

function convertdatapoints(datapoints::Array{Pair{String,Int64}}; state::Bool=false)
  nshots = size(datapoints)[1]
  newdata = Matrix{String}(undef, nshots, size(datapoints)[2])

  for n in 1:nshots
    newdata[n, :] = convertdatapoint(datapoints[n, :]; state=state)
  end
  return newdata
end

function convertdatapoints(outcome::Matrix{Int64}, basis::Matrix{String}; state::Bool=false)
  nshots = size(datapoints)[1]
  newdata = Matrix{String}(undef, nshots, size(datapoints)[2])

  for n in 1:nshots
    newdata[n, :] = convertdatapoint(datapoints[n, :]; state=state)
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
function split_dataset(data::Matrix; train_ratio::Float64=0.9, randomize::Bool=true)
  ndata = size(data, 1)
  ntrain = Int(ndata * train_ratio)
  ntest = ndata - ntrain
  if randomize
    data = data[shuffle(1:end), :]
  end
  train_data = data[1:ntrain, :]
  test_data = data[(ntrain + 1):end, :]
  return train_data, test_data
end

"""
    ischoi(M::LPDO)

Check whether a given LPDO{MPO}  
"""
function ischoi(M::LPDO{MPO})
  !haschoitags(M) && return false
  for j in 1:length(M)
    length(inds(M.X[j], "Site")) != 2 && return false
  end
  return true
end

function ischoi(M::MPO) 
  !haschoitags(M) && return false
  for j in 1:length(M)
    length(inds(M[j], "Site")) != 4 && return false
  end
  return true
end

ischoi(M::ITensor) = haschoitags(M)


"""
    haschoitags(L::LPDO)
    haschoitags(M::Union{MPS,MPO})

Check whether the TN has input/output Choi tags
"""
function haschoitags(L::LPDO)
  for j in 1:length(L)
    !(hastags(L.X[j], "Input") && hastags(L.X[j], "Output")) && return false
  end
  return true
end
function haschoitags(M::Union{MPS,MPO})
  for j in 1:length(M)
    !(hastags(M[j], "Input") && hastags(M[j], "Output")) && return false
  end
  return true
end

function haschoitags(M::ITensor)
  N = nsites(M)
  for j in 1:N
    !(hastags(M,"Input,n=$j") && hastags(M,"Output,n=$j")) && return false
  end
  return true
end

"""
    choitags(U::MPO)

Assign the input/output tags defined for a Choi matrix to an MPO.

  σ₁ -o- σ₁′       σ₁ⁱ -o- σ₁ᴼ   
      |                 |
  σ₂ -o- σ₂′  ⟶    σ₂ⁱ -o- σ₂ᴼ
      |                 |
  σ₃ -o- σ₃′       σ₃ⁱ -o- σ₃ᴼ
                  
"""
function choitags(U::MPO)
  haschoitags(U) && return U
  U = addtags(siteinds, U, "Input"; plev=0)
  U = addtags(siteinds, U, "Output"; plev=1)
  return noprime(siteinds, U)
end

function choitags(T::ITensor)
  haschoitags(T) && return T
  T = addtags(T, "Input"; plev=0)
  T = addtags(T, "Output"; plev=1)
  return noprime(T)
end

choitags(O::ITensorOperator) = 
  ITensorState(choitags(O.T))

"""
    mpotags(U::MPO)

Inverse of choitags.
"""
function mpotags(U::MPO)
  U = prime(U; tags="Output")
  U = removetags(U, "Input")
  U = removetags(U, "Output")
  return U
end

mpotags(M::Union{MPS,MPO}) = mpotags(LPDO(M)).X

"""
    unitary_mpo_to_choi_mps(U::MPO)


     MPO          MPS (vectorized MPO) 
  σ₁ -o- σ₁′       o= (σ₁ⁱ,σ₁ᴼ)   
      |            | 
  σ₂ -o- σ₂′  ⟶    o= (σ₂ⁱ,σ₂′ᴼ)
      |            | 
  σ₃ -o- σ₃′       o= (σ₃ⁱ,σ₃′ᴼ)
                  
Transforms a unitary MPO into a Choi MPS with appropriate tags.
"""
unitary_mpo_to_choi_mps(U::MPO) = convert(MPS, choitags(U))
unitary_mpo_to_choi_mps(T::ITensor) = choitags(T) 
unitary_mpo_to_choi_mps(L::LPDO{MPO}) = unitary_mpo_to_choi_mps(L.X)


"""
    unitary_mpo_to_choi_mpo(U::MPO)


     MPO                   MPO
  σ₁ -o- σ₁′     (σ₁ⁱ,σ₁ᴼ) =o= (σ₁′ⁱ,σ₁′ᴼ)   
      |                     | 
  σ₂ -o- σ₂′  ⟶  (σ₂ⁱ,σ₂ᴼ) =o= (σ₂′ⁱ,σ₂′ᴼ)
      |                     | 
  σ₃ -o- σ₃′     (σ₃ⁱ,σ₃ᴼ) =o= (σ₃′ⁱ,σ₃′ᴼ)
                  
Convert a unitary MPO to a Choi matrix represented as an MPO with 4 site indices.
"""
unitary_mpo_to_choi_mpo(U::MPO) = MPO(LPDO(convert(MPS, choitags(U))))

"""
    choi_mps_to_unitary_mpo(Ψ::MPS)

Transforms a Choi MPS into an MPO with appropriate tags.
Inverse of `unitary_mpo_to_choi_mps`.
"""
choi_mps_to_unitary_mpo(Ψ::MPS) = mpotags(convert(MPO, Ψ))
choi_mps_to_unitary_mpo(L::LPDO{MPS}) = choi_mps_to_unitary_mpo(L.X)


function nsites(T::ITensor)
  s1 = inds(T,tags="Site,n=1")
  # Wavefunction
  if length(s1) == 1 || length(s1) == 2
    return length(inds(T,plev=0))
  # Choi matrix
  elseif length(s1) == 4
    return length(inds(T,plev=0)) ÷ 2
  else
    error("Indices not recognized")
  end
end

nsites(M::Union{MPS,MPO,LPDO}) = 
  length(M)

