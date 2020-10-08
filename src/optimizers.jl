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
function SGD(;η::Float64=0.01,γ::Float64=0.0)
  v = ITensor[]
  return SGD(η,γ,v)
end

Base.copy(opt::SGD) = SGD(opt.η, opt.γ, copy(opt.v))

#SGD(M::Union{MPS,MPO};η::Float64=0.01,γ::Float64=0.0) = SGD(LPDO(M);η=η,γ=γ)

"""
    update!(L::LPDO,∇::Array,opt::SGD; kwargs...)

Update parameters with SGD.

1. `vⱼ = γ * vⱼ - η * ∇ⱼ`: integrated velocity
2. `θⱼ = θⱼ + vⱼ`: parameter update
"""
function update!(L::LPDO,∇::Array,opt::SGD; kwargs...)
  M = L.X
  if isempty(opt.v)
    for j in 1:length(M)
      push!(opt.v,ITensor(zeros(size(M[j])),inds(M[j])))
    end
  end
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

Base.copy(opt::AdaGrad) = AdaGrad(opt.η, opt.ϵ, copy(opt.∇²))

"""
    AdaGrad(L::LPDO;η::Float64=0.01,ϵ::Float64=1E-8)


# Parameters
  - `η`: learning rate
  - `ϵ`: shift 
  - `∇²`: square gradients (running sums)
"""
function AdaGrad(;η::Float64=0.01,ϵ::Float64=1E-8)
  ∇² = ITensor[]
  return AdaGrad(η,ϵ,∇²)
end

#AdaGrad(ψ::Union{MPS,MPO};η::Float64=0.01,ϵ::Float64=1E-8) = AdaGrad(LPDO(ψ);η=η,ϵ=ϵ)

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
  if isempty(opt.∇²)
    for j in 1:length(M)
      push!(opt.∇²,ITensor(zeros(size(M[j])),inds(M[j])))
    end
  end
  for j in 1:length(M)
    #opt.∇²[j] += ∇[j] .^ 2 
    Re∇ = real.(∇[j])
    Im∇ = imag.(∇[j])
    opt.∇²[j] += Re∇ .^ 2
    opt.∇²[j] += im * Im∇ .^ 2
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

Base.copy(opt::AdaDelta) = AdaDelta(opt.γ, opt.ϵ, copy(opt.∇²), copy(opt.Δθ²))

"""
    AdaDelta(L::LPDO;γ::Float64=0.9,ϵ::Float64=1E-8)

# Parameters
  - `γ`: friction coefficient
  - `ϵ`: shift 
  - `∇²`: square gradients (decaying average)
  - `Δθ²`: square updates (decaying average)
"""
function AdaDelta(;γ::Float64=0.9,ϵ::Float64=1E-8)
  Δθ² = ITensor[]
  ∇² = ITensor[]
  return AdaDelta(γ,ϵ,∇²,Δθ²)
end

#AdaDelta(ψ::Union{MPS,MPO};γ::Float64=0.9,ϵ::Float64=1E-8) = AdaDelta(LPDO(ψ);γ=γ,ϵ=ϵ) 

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
  if isempty(opt.∇²)
    for j in 1:length(M)
      push!(opt.Δθ²,ITensor(zeros(size(M[j])),inds(M[j])))
      push!(opt.∇²,ITensor(zeros(size(M[j])),inds(M[j])))
    end
  end
  for j in 1:length(M)
    # Update square gradients
    Re∇ = real.(∇[j])
    Im∇ = imag.(∇[j])
    
    opt.∇²[j] =  opt.γ * opt.∇²[j]
    opt.∇²[j] += (1-opt.γ) * Re∇ .^2
    opt.∇²[j] += im * (1-opt.γ) * Im∇ .^2
    #opt.∇²[j] = opt.γ * opt.∇²[j] + (1-opt.γ) * ∇[j] .^ 2
    
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
    ReΔθ = real.(Δθ)
    ImΔθ = imag.(Δθ)
    opt.Δθ²[j] =  opt.γ * opt.Δθ²[j]
    opt.Δθ²[j] += (1-opt.γ) * ReΔθ .^2
    opt.Δθ²[j] += im * (1-opt.γ) * ImΔθ .^2
    #opt.Δθ²[j] = opt.γ * opt.Δθ²[j] + (1-opt.γ) * Δθ .^ 2
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

Base.copy(opt::Adam) = Adam(opt.η, opt.β₁, opt.β₂, opt.ϵ, copy(opt.∇), copy(opt.∇²))

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
function Adam(;η::Float64=0.001,
              β₁::Float64=0.9,β₂::Float64=0.999,ϵ::Float64=1E-7)
  ∇ = ITensor[]
  ∇² = ITensor[]
  return Adam(η,β₁,β₂,ϵ,∇,∇²)
end

#Adam(ψ::Union{MPS,MPO};η::Float64=0.001,β₁::Float64=0.9,β₂::Float64=0.999,ϵ::Float64=1E-7) = Adam(LPDO(ψ);η=η,β₁=β₁,β₂=β₂,ϵ=ϵ)

"""
    update!(L::LPDO,∇::Array,opt::Adam; kwargs...)

    update!(ψ::MPS,∇::A0rray,opt::Adam; kwargs...)

Update parameters with Adam
"""
function update!(L::LPDO,∇::Array,opt::Adam; kwargs...)
  M = L.X
  if isempty(opt.∇²)
    for j in 1:length(M)
      push!(opt.∇,ITensor(zeros(size(M[j])),inds(M[j])))
      push!(opt.∇²,ITensor(zeros(size(M[j])),inds(M[j])))
    end
  end
  t = kwargs[:step]
  for j in 1:length(M)
    # Update square gradients
    opt.∇[j]  = opt.β₁ * opt.∇[j]  + (1-opt.β₁) * ∇[j]
    
    Re∇ = real.(∇[j])
    Im∇ = imag.(∇[j])
    opt.∇²[j] =  opt.β₂ * opt.∇²[j]
    opt.∇²[j] += (1-opt.β₂) * Re∇ .^2
    opt.∇²[j] += im * (1-opt.β₂) * Im∇ .^2
    #opt.∇²[j] = opt.β₂ * opt.∇²[j] + (1-opt.β₂) * ∇[j] .^ 2
    
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

update!(ψ::MPS,∇::Array,opt::Adam; kwargs...) =
  update!(LPDO(ψ),∇,opt; kwargs...)

