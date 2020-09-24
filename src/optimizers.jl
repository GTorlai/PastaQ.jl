abstract type Optimizer end

struct SGD <: Optimizer 
  η::Float64
  γ::Float64
  v::Vector{ITensor}
end

""" 
    SGD(L::LPDO;η::Float64=0.01,γ::Float64=0.0)

Stochastic gradient descent with momentum.

# Parameters
  - `η`: learning rate
  - `γ`: friction coefficient
  - `v`: "velocity"
"""
function SGD(L::LPDO;η::Float64=0.01,γ::Float64=0.0)
  M = L.X
  v = ITensor[]
  for j in 1:length(L)
    push!(v,ITensor(zeros(size(M[j])),inds(M[j])))
  end
  return SGD(η,γ,v)
end

SGD(ψ::MPS;η::Float64=0.01,γ::Float64=0.0) = SGD(LPDO(ψ);η=η,γ=γ)

"""
    update!(L::LPDO,∇::Array,opt::SGD; kwargs...)

Update parameters with SGD.

1. `vⱼ = γ * vⱼ - η * ∇ⱼ`: integrated velocity
2. `θⱼ = θⱼ + vⱼ`: parameter update
"""
function update!(L::LPDO,∇::Array,opt::SGD; kwargs...)
  M = L.X
  for j in 1:length(M)
    opt.v[j] = opt.γ * opt.v[j] - opt.η * ∇[j]
    M[j] = M[j] + opt.v[j] 
  end
end

update!(ψ::MPS,∇::Array,opt::SGD; kwargs...) = update!(LPDO(ψ),∇,opt;kwargs...)



struct AdaGrad <: Optimizer 
  η::Float64
  ϵ::Float64
  ∇²::Vector{ITensor}
end

"""
    AdaGrad(L::LPDO;η::Float64=0.01,ϵ::Float64=1E-8)


# Parameters
  - `η`: learning rate
  - `ϵ`: shift 
  - `∇²`: square gradients (running sums)
"""
function AdaGrad(L::LPDO;η::Float64=0.01,ϵ::Float64=1E-8)
  M = L.X
  ∇² = ITensor[]
  for j in 1:length(M)
    push!(∇²,ITensor(zeros(size(M[j])),inds(M[j])))
  end
  return AdaGrad(η,ϵ,∇²)
end

AdaGrad(ψ::MPS;η::Float64=0.01,ϵ::Float64=1E-8) = AdaGrad(LPDO(ψ);η=η,ϵ=ϵ)

"""
    update!(L::LPDO,∇::Array,opt::AdaGrad; kwargs...)

    update!(ψ::MPS,∇::Array,opt::AdaGrad; kwargs...)

Update parameters with AdaGrad.

1. `gⱼ += ∇ⱼ²`: running some of square gradients
2. `Δθⱼ = η * ∇ⱼ / (sqrt(gⱼ+ϵ)` 
2. `θⱼ = θⱼ - Δθⱼ`: parameter update
"""
function update!(L::LPDO,∇::Array,opt::AdaGrad; kwargs...)
  M = L.X
  for j in 1:length(M)
    opt.∇²[j] += ∇[j] .^ 2 
    ∇² = copy(opt.∇²[j])
    ∇² .+= opt.ϵ
    g = sqrt.(∇²)    
    δ = g .^ -1             
    M[j] = M[j] - opt.η * (noprime(∇[j]) ⊙ δ)
  end
end

update!(ψ::MPS,∇::Array,opt::AdaGrad; kwargs...) = update!(LPDO(ψ),∇,opt; kwargs...)





struct AdaDelta <: Optimizer 
  γ::Float64
  ϵ::Float64
  ∇²::Vector{ITensor}
  Δθ²::Vector{ITensor}
end

"""
    AdaDelta(L::LPDO;γ::Float64=0.9,ϵ::Float64=1E-8)

# Parameters
  - `γ`: friction coefficient
  - `ϵ`: shift 
  - `∇²`: square gradients (decaying average)
  - `Δθ²`: square updates (decaying average)
"""
function AdaDelta(L::LPDO;γ::Float64=0.9,ϵ::Float64=1E-8)
  M = L.X
  Δθ² = ITensor[]
  ∇² = ITensor[]
  for j in 1:length(M)
    push!(Δθ²,ITensor(zeros(size(M[j])),inds(M[j])))
    push!(∇²,ITensor(zeros(size(M[j])),inds(M[j])))
  end
  return AdaDelta(γ,ϵ,∇²,Δθ²)
end

AdaDelta(ψ::MPS;γ::Float64=0.9,ϵ::Float64=1E-8) = AdaDelta(LPDO(ψ);γ=γ,ϵ=ϵ) 

"""
    update!(L::LPDO,∇::Array,opt::AdaDelta; kwargs...)
    
    update!(ψ::MPS,∇::Array,opt::AdaDelta; kwargs...)

Update parameters with AdaDelta


1. `gⱼ = γ * gⱼ + (1-γ) * ∇ⱼ²`: decaying average
2. `Δθⱼ = ∇ⱼ * sqrt(pⱼ) / sqrt(gⱼ+ϵ) ` 
3. `θⱼ = θⱼ - Δθⱼ`: parameter update
4. `pⱼ = γ * pⱼ + (1-γ) * Δθⱼ²`: decaying average
"""
function update!(L::LPDO,∇::Array,opt::AdaDelta; kwargs...)
  M = L.X
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

update!(ψ::MPS,∇::Array,opt::AdaDelta; kwargs...) = update!(LPDO(ψ),∇,opt; kwargs...)

struct Adam <: Optimizer 
  η::Float64
  β₁::Float64
  β₂::Float64
  ϵ::Float64
  ∇::Vector{ITensor}    # m in the paper
  ∇²::Vector{ITensor}   # v in the paper
end

"""
    Adam(L::LPDO;η::Float64=0.001,
         β₁::Float64=0.9,β₂::Float64=0.999,ϵ::Float64=1E-7)

    Adam(ψ::MPS;η::Float64=0.001,
         β₁::Float64=0.9,β₂::Float64=0.999,ϵ::Float64=1E-7)

# Parameters
  - `η`: learning rate
  - `β₁`: decay rate 1 
  - `β₂`: decay rate 2
  - `ϵ`: shift 
  - `∇`: gradients (decaying average)
  - `∇²`: square gradients (decaying average)
"""
function Adam(L::LPDO;η::Float64=0.001,
              β₁::Float64=0.9,β₂::Float64=0.999,ϵ::Float64=1E-7)
  M = L.X
  ∇ = ITensor[]
  ∇² = ITensor[]
  for j in 1:length(M)
    push!(∇,ITensor(zeros(size(M[j])),inds(M[j])))
    push!(∇²,ITensor(zeros(size(M[j])),inds(M[j])))
  end
  return Adam(η,β₁,β₂,ϵ,∇,∇²)
end

Adam(ψ::MPS;η::Float64=0.001,β₁::Float64=0.9,β₂::Float64=0.999,ϵ::Float64=1E-7) = Adam(LPDO(ψ);η=η,β₁=β₁,β₂=β₂,ϵ=ϵ)

"""
    update!(L::LPDO,∇::Array,opt::Adam; kwargs...)

    update!(ψ::MPS,∇::A0rray,opt::Adam; kwargs...)

Update parameters with Adam
"""
function update!(L::LPDO,∇::Array,opt::Adam; kwargs...)
  M = L.X
  t = kwargs[:step]
  for j in 1:length(M)
    # Update square gradients
    opt.∇[j]  = opt.β₁ * opt.∇[j]  + (1-opt.β₁) * ∇[j]
    opt.∇²[j] = opt.β₂ * opt.∇²[j] + (1-opt.β₂) * ∇[j] .^ 2
    
    g1 = opt.∇[j]  ./ (1-opt.β₁^t)
    g2 = opt.∇²[j] ./ (1-opt.β₂^t)
    
    den = sqrt.(g2) 
    den .+= opt.ϵ
    δ = den .^-1
    Δθ = g1 ⊙ δ
    
    # Update parameters
    M[j] = M[j] - opt.η * Δθ
  end
end

update!(ψ::MPS,∇::Array,opt::Adam; kwargs...) = update!(LPDO(ψ),∇,opt; kwargs...)

#struct AdaMax <: Optimizer 
#  η::Float64
#  β₁::Float64
#  β₂::Float64
#  ∇::Vector{ITensor}    # m in the paper
#  u::Vector{ITensor}   # v in the paper
#end
#
#function AdaMax(M::Union{MPS,MPO};η::Float64=0.001,β₁::Float64=0.9,β₂::Float64=0.999)
#  ∇ = ITensor[]
#  u = ITensor[]
#  for j in 1:length(M)
#    push!(∇,ITensor(zeros(size(M[j])),inds(M[j])))
#    push!(u,ITensor(zeros(size(M[j])),inds(M[j])))
#  end
#  return AdaMax(η,β₁,β₂,∇,u)
#end
#
#function update!(M::Union{MPS,MPO},∇::Array,opt::AdaMax; kwargs...)
#  t = kwargs[:step]
#  for j in 1:length(M)
#    # Update square gradients
#    opt.∇[j]  = opt.β₁ * opt.∇[j]  + (1-opt.β₁) * ∇[j]
#    opt.u[j]  = max.(opt.β₂ * opt.u[j], abs.(opt.∇[j])) 
#    δ = opt.u[j] .^-1
#    Δθ = opt.∇[j] ⊙ δ
#    # Update parameters
#    M[j] = M[j] - (opt.η/(1-opt.β₁^t)) * Δθ
#  end
#end

