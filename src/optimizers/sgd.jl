abstract type Optimizer end

struct Sgd <: Optimizer 
  η::Float64
end

function Sgd(;η::Float64=0.01)
  return Sgd(η)
end

function update!(M::Union{MPS,MPO},G::Array{ITensor},opt::Sgd)
  for j in 1:length(M)
    M[j] = M[j] - opt.η * noprime(G[j])
  end
end
