"""
    array(M::MPS; reverse::Bool = true)

Generate the full dense vector from an MPS
"""
function array(M::MPS; reverse::Bool=true)
  # check if it is a vectorized MPO
  if length(siteinds(M, 1)) == 2
    s = []
    for j in 1:length(M)
      push!(s, firstind(M[j]; tags="Input"))
      push!(s, firstind(M[j]; tags="Output"))
    end
  else
    s = siteinds(M)
  end
  if reverse
    s = Base.reverse(s)
  end
  C = combiner(s...)
  Mvec = prod(M) * dag(C)
  return ITensors.array(Mvec)
end

"""
    array(M::MPO; reverse::Bool = true)
    array(L::LPDO; reverse::Bool = true)

Generate the full dense matrix from an MPO or LPDO.
"""
function array(M::MPO; reverse::Bool=true)
  if length(siteinds(M, 1)) == 4
    s = []
    for j in 1:length(M)
      push!(s, firstind(M[j]; tags="Input", plev=0))
      push!(s, firstind(M[j]; tags="Output", plev=0))
    end
  else
    s = firstsiteinds(M; plev=0)
  end
  if reverse
    s = Base.reverse(s)
  end
  C = combiner(s...)
  Mmat = prod(M) * dag(C) * C'
  c = combinedind(C)
  return ITensors.array(permute(Mmat, c', c))
 
end

function array(L::LPDO{MPO}; reverse::Bool=true)
  !ischoi(L) && return PastaQ.array(MPO(L); reverse=reverse)
  M = MPO(L)
  s = []
  for j in 1:length(M)
    push!(s, firstind(M[j]; tags="Input", plev=0))
    push!(s, firstind(M[j]; tags="Output", plev=0))
  end
  if reverse
    s = Base.reverse(s)
  end
  C = combiner(s...)
  Mmat = prod(M) * dag(C) * C'
  c = combinedind(C)
  return ITensors.array(permute(Mmat, c', c))
end


is_operator(T::ITensor) = !isempty(inds(T,tags="Site,n=1",plev=1))

array(T::ITensor; reverse::Bool = true) = 
  (is_operator(T) ? tomatrix(T) : tovector(T))

function tovector(M::ITensor)
  if length(inds(M,tags="n=1")) > 1
    error("Cannot transform a density matrix into a vector")
  end
  length(inds(M)) == 1 && return ITensors.array(M)
  s = []
  for j in 1:length(inds(M,plev=0))
    push!(s,firstind(M,tags="n=$(j)",plev=0))
  end
  s = Base.reverse(s)
  C = combiner(s...)
  return ITensors.array(M * C)
end

function tomatrix(M::ITensor)
  if length(inds(M,tags="n=1")) == 1
    error("Cannot transform a wavefunctionm into a matrix")
  end
  length(inds(M)) == 2 && return ITensors.array(M)
  s = []
  for j in 1:length(inds(M,plev=0))
    push!(s,firstind(M,tags="n=$(j)",plev=0))
  end
  s = Base.reverse(s)
  C = combiner(s...)
  Mmat = M * dag(C) * C'
  c = combinedind(C)
  return ITensors.array(permute(Mmat, c', c))
end

