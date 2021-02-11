"""
    array(M::MPS; reverse::Bool = true)

Generate the full dense vector from an MPS
"""
function array(M::MPS; reverse::Bool = true)
  # check if it is a vectorized MPO
  if length(siteinds(M,1)) == 2
    s = []
    for j in 1:length(M)
      push!(s, firstind(M[j], tags="Input"))
      push!(s, firstind(M[j], tags="Output"))
    end
  else
    s = siteinds(M)
  end 
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
  if length(siteinds(M,1)) == 4
    s = []
    for j in 1:length(M)
      push!(s, firstind(M[j], tags="Input", plev =0))
      push!(s, firstind(M[j], tags="Output", plev = 0))
    end
  else
    s = firstsiteinds(M; plev = 0)
  end 
  if reverse
    s = Base.reverse(s)
  end
  C = combiner(s...)
  Mmat = prod(M) * dag(C) * C'
  c = combinedind(C)
  return array(permute(Mmat, c', c))
end

function array(L::LPDO{MPO}; reverse::Bool = true) 
  !ischoi(L) && return array(MPO(L); reverse = reverse)
  M = MPO(L) 
  s = []
  for j in 1:length(M)
    push!(s, firstind(M[j], tags="Input", plev=0))
    push!(s, firstind(M[j], tags="Output", plev=0))
  end
  if reverse
    s = Base.reverse(s)
  end
  C = combiner(s...)
  Mmat = prod(M) * dag(C) * C'
  c = combinedind(C)
  return array(permute(Mmat, c', c))
end

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

"""
    _ischoi(M::LPDO)

Check whether a given LPDO{MPO}  
"""
ischoi(M::LPDO{MPO}) = (length(inds(M.X[1],"Site")) == 2 && haschoitags(M))#hastags(M.X[1],"Purifier"))

ischoi(M::MPO) = (length(inds(M[1],"Site")) == 4 && haschoitags(M))



"""
    haschoitags(L::LPDO)

    haschoitags(M::Union{MPS,MPO})

Check whether the TN has input/output Choi tags
"""
haschoitags(L::LPDO) = (hastags(inds(L.X[1]),"Input") && hastags(inds(L.X[1]),"Output"))
haschoitags(M::Union{MPS,MPO}) = haschoitags(LPDO(M)) 


"""
    choitags(U::Union{MPS, MPO})
    choitags(L::LPDO{MPO})

Assign the input/output tags defined for a Choi matrix to an MPO.

  σ₁ -o- σ₁′       σ₁ⁱ -o- σ₁ᴼ   
      |                 |
  σ₂ -o- σ₂′  ⟶    σ₂ⁱ -o- σ₂ᴼ
      |                 |
  σ₃ -o- σ₃′       σ₃ⁱ -o- σ₃ᴼ
                  
"""
function choitags(L::LPDO{MPO})
  haschoitags(L) && return L
  U = copy(L.X)
  U = addtags(U, "Input", plev = 0, tags = "Site")
  U = addtags(U, "Output", plev = 1, tags = "Site")
  return LPDO(noprime(U))
end

choitags(U::MPO) = choitags(LPDO(U)).X

"""
    _mpotags(Ψ::MPS)

Inverse of _choitags
"""

function mpotags(L::LPDO)
  !haschoitags(L) && return L
  U = copy(L.X) 
  U = prime(U,tags="Output")
  U = removetags(U, "Input")
  U = removetags(U, "Output")
  return LPDO(U)
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
    _choiMPS_to_unitaryMPO(Ψ::MPS)

Transforms a Choi MPS into an MPO with appropriate tags.
Inverse of `_unitaryMPO_to_choiMPS`.
"""
#_choiMPS_to_unitaryMPO(L::LPDO{MPS}) = LPDO(convert(MPO, _mpotags(L).X))
#_choiMPS_to_unitaryMPO(Ψ::MPS)       = _choiMPS_to_unitaryMPO(LPDO(Ψ)).X
choi_mps_to_unitary_mpo(Ψ::MPS)       = convert(MPO, mpotags(Ψ))
choi_mps_to_unitary_mpo(L::LPDO{MPS}) = choi_mps_to_unitary_mpo(L.X) 

function nqubits(g::Tuple)
  s = g[2]
  n = (s isa Number ? s : maximum(s))
  return n
end

nqubits(gates::Vector{<:Any}) = maximum((nqubits(gate) for gate in gates))


nlayers(circuit::Vector{<:Any}) = 1
nlayers(circuit::Vector{<:Vector{<:Any}}) = length(circuit)

ngates(circuit::Vector{<:Any}) = length(circuit)
ngates(circuit::Vector{<:Vector{<:Any}}) = length(vcat(circuit...))

