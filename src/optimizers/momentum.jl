struct Momentum <: Optimizer 
  η::Float64
  γ::Float64
end

function Momentum(;η::Float64=0.01,γ::Float64=0.9)
  return Momentum(η,γ)
end

function update!(M::Union{MPS,MPO},G::Array{ITensor},opt::Momentum)
  for j in 1:length(M)
    new_G = opt.γ * M[j] + noprime(G[j])#(1.0-opt.γ) * noprime(G[j])
    M[j] = M[j] - opt.η * new_G
  end
end
