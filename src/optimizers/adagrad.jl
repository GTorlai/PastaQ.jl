
struct Adagrad <: Optimizer 
  η::Float64
  ϵ::Float64
  ∇²::Vector{ITensor}
end

function Adagrad(M::Union{MPS,MPO};η::Float64=0.01,ϵ::Float64=1E-8)
  ∇² = ITensor[]
  for j in 1:length(M)
    push!(∇²,ITensor(zeros(size(M[j])),inds(M[j])))
  end
  return Adagrad(η,ϵ,∇²)
end

#function update!(M::Union{MPS,MPO},∇::Array{ITensor},opt::Adagrad)
function update!(M::Union{MPS,MPO},∇,opt::Adagrad)
  for j in 1:length(M)
    opt.∇²[j] += ∇[j] .^ 2
    opt.∇²[j] .+= opt.ϵ 
    g = sqrt.(opt.∇²[j])
    δ = g .^ -1
    #TODO DOUBLE CHECK THIS ABOVE
    M[j] = M[j] - opt.η * (noprime(∇[j]) .* prime(δ))
  end
end
