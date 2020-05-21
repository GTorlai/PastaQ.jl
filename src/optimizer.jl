struct Optimizer
  η::Float64
end

function Optimizer(;η::Float64=0.01)
  return Optimizer(η)
end

function updateSGD!(M::Union{MPS,MPO},G::Array{ITensor},opt::Optimizer)
  for j in 1:length(M)
    M[j] = M[j] - opt.η * noprime(G[j])
  end
end
