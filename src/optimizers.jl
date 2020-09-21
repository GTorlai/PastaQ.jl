abstract type Optimizer end

"""
SGD
"""

struct SGD <: Optimizer 
  η::Float64
end

function SGD(;η::Float64=0.01)
  return SGD(η)
end

function SGD(M::Union{MPS,MPO};η::Float64=0.01)
  return SGD(η)
end

#function update!(M::Union{MPS,MPO},G::Array{ITensor},opt::SGD)
function update!(M::Union{MPS,MPO},∇::Array,opt::SGD)
  for j in 1:length(M)
    M[j] = M[j] - opt.η * noprime(∇[j])
  end
end


"""
MOMENTUM
"""

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


"""
ADAGRAD
"""

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
function update!(M::Union{MPS,MPO},∇::Array,opt::Adagrad)
  for j in 1:length(M)
    opt.∇²[j] += ∇[j] .^ 2 
    ∇² = copy(opt.∇²[j])
    ∇² .+= opt.ϵ
    g = sqrt.(∇²)    
    δ = g .^ -1             
    M[j] = M[j] - opt.η * (noprime(∇[j]) ⊙ δ)
  end
end


"""
ADADELTA
"""

struct Adadelta <: Optimizer 
  γ::Float64
  ϵ::Float64
  ∇²::Vector{ITensor}
  Δθ²::Vector{ITensor}
end

function Adadelta(M::Union{MPS,MPO};γ::Float64=0.9,ϵ::Float64=1E-8)
  Δθ² = ITensor[]
  ∇² = ITensor[]
  for j in 1:length(M)
    push!(Δθ²,ITensor(zeros(size(M[j])),inds(M[j])))
    push!(∇²,ITensor(zeros(size(M[j])),inds(M[j])))
  end
  return Adadelta(γ,ϵ,∇²,Δθ²)
end

function update!(M::Union{MPS,MPO},∇,opt::Adadelta)
  for j in 1:length(M)
    # Update square gradients
    opt.∇²[j] = opt.γ * opt.∇²[j] + (1-opt.γ) * ∇[j] .^ 2
    
    # Get RMS signal for square gradients
    ∇² = copy(opt.∇²[j])
    ∇² .+= opt.ϵ
    g1 = sqrt.(∇²)    
    δ1 = g1 .^(-1)

    # Get RMS signal for square updates
    Δθ² = copy(opt.Δθ²[j])
    Δθ² .+= opt.ϵ
    g2 = sqrt.(Δθ²)
    #g2 = sqrt.(opt.Δθ²[j] .+ opt.ϵ)
    Δ = noprime(∇[j]) ⊙ δ1
    Δθ = noprime(Δ) ⊙ g2

    ## Update parameters
    M[j] = M[j] - Δθ

    # Update square updates
    opt.Δθ²[j] = opt.γ * opt.Δθ²[j] + (1-opt.γ) * Δθ .^ 2
  end
end
