
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
    opt.∇²[j] .= opt.γ * opt.∇²[j] + (1-opt.γ) * ∇[j] .^ 2
    
    # Get RMS signal for square gradients
    #opt.∇²[j] .+= opt.ϵ 
    g1 = sqrt.(opt.∇²[j] .+= opt.ϵ)
    κ1 = g1 .^ -1
    
    # Get RMS signal for square updates
    g2 = sqrt.(opt.Δθ²[j] .+= opt.ϵ)
    @show g2 
    Δ = noprime(∇[j]) .* prime(κ1)
    Δθ = noprime(Δ) .* prime(g2)

    # Update parameters
    M[j] = M[j] - Δθ

    # Update square updates
    opt.Δθ²[j] .= opt.γ * opt.Δθ²[j] + (1-opt.γ) * Δθ .^ 2
  end
end
