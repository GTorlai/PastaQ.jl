abstract type Optimizer end

struct SGD <: Optimizer 
  η::Float64
end

function SGD(;η::Float64=0.01)
  return SGD(η)
end

#function update!(M::Union{MPS,MPO},G::Array{ITensor},opt::SGD)
function update!(M::Union{MPS,MPO},G::Array,opt::SGD)
  for j in 1:length(M)
    M[j] = M[j] - opt.η * noprime(G[j])
  end
end
