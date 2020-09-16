
struct Momentum <: Optimizer 
  η::Float64
  μ::Float64
  z::Vector{ITensor}
end

function Momentum(M::Union{MPS,MPO};η::Float64=0.01,μ::Float64=0.9)
  z = ITensor[]
  for j in 1:length(M)
    push!(z,ITensor(zeros(size(M[j])),inds(M[j])))
  end
  return Momentum(η,μ,z)
end

function update!(M::Union{MPS,MPO},∇::Array{ITensor},opt::Momentum)
  for j in 1:length(M)
    opt.z[j] = opt.μ * opt.z[j] + noprime(∇[j])
    M[j] = M[j] - opt.η * opt.z[j]
  end
end
