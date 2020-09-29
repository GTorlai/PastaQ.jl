using PastaQ
using ITensors
using Test
using Random

function flatten_tensorarray(T0)
  T = copy(T0)
  flatT = reshape(array(T[1]),(1,prod(size(T[1]))))
  for j in 2:length(T)
    tmp = reshape(array(T[j]),(1,prod(size(T[j]))))
    flatT = hcat(flatT,tmp)
  end
  return flatT
end

function generategradients(sites,links,χ,d)
  σ = 1e-1
  N = length(sites)
  ∇ = ITensor[] 
  rand_mat = σ * (ones(d,χ) - 2*rand(d,χ))
  rand_mat += im * σ * (ones(d,χ) - 2*rand(d,χ))
  push!(∇,ITensor(rand_mat,sites[1],links[1]))
  for j in 2:N-1
    rand_mat = σ * (ones(χ,d,χ) - 2*rand(χ,d,χ))
    rand_mat += im * σ * (ones(χ,d,χ) - 2*rand(χ,d,χ))
    push!(∇,ITensor(rand_mat,links[j-1],sites[j],links[j]))
  end
  # Site N
  rand_mat = σ * (ones(χ,d) - 2*rand(χ,d))
  rand_mat += im * σ * (ones(χ,d) - 2*rand(χ,d))
  push!(∇,ITensor(rand_mat,links[N-1],sites[N]))
  return ∇ 
end
  
function generategradients(sites,links,kraus,χ,ξ,d)
  σ = 1e-1
  N = length(sites)
  ∇ = ITensor[] 
  rand_mat = σ * (ones(d,χ,ξ) - 2*rand(d,χ,ξ))
  rand_mat += im * σ * (ones(d,χ,ξ) - 2*rand(d,χ,ξ))
  push!(∇,ITensor(rand_mat,sites[1],links[1],kraus[1]))
  # Site 2..N-1
  for j in 2:N-1
    rand_mat = σ * (ones(d,χ,ξ,χ) - 2*rand(d,χ,ξ,χ))
    rand_mat += im * σ * (ones(d,χ,ξ,χ) - 2*rand(d,χ,ξ,χ))
    push!(∇,ITensor(rand_mat,sites[j],links[j-1],kraus[j],links[j]))
  end
  # Site N
  rand_mat = σ * (ones(d,χ,ξ) - 2*rand(d,χ,ξ))
  rand_mat += im * σ * (ones(d,χ,ξ) - 2*rand(d,χ,ξ))
  push!(∇,ITensor(rand_mat,sites[N],links[N-1],kraus[N]))
  return ∇ 
end
  


@testset "sgd" begin
  N = 3
  χ = 4
  d = 2
  ψ = randomstate(N;χ=χ,σ=1.0)
  sites = siteinds(ψ) 
  links = linkinds(ψ)

  ψ_flat  = flatten_tensorarray(ψ)
  v_flat = zeros(size(ψ_flat))
  
  η = 0.1
  γ = 0.9
  opt = SGD(η=η,γ=γ)
  
  for n in 1:100
    ∇ = generategradients(sites,links,χ,d)
    ∇_flat = flatten_tensorarray(∇)
    ψ_flat  = flatten_tensorarray(ψ)
    # algorithm
    update!(ψ,∇,opt)
    ψ′_alg_flat = flatten_tensorarray(ψ)
    # exact
    v_flat = γ * v_flat - η * ∇_flat 
    ψ′_flat = ψ_flat + v_flat 
    @test ψ′_flat ≈ ψ′_alg_flat rtol = 1e-4
  end
  
  N = 3
  χ = 4
  ξ = 3
  d = 2
  ρ = randomstate(N;mixed=true,χ=χ,ξ=ξ,σ=1.0)
  sites = firstsiteinds(ρ.X) 
  links = linkinds(ρ.X)
  kraus = [firstind(ρ.X[j],tags="Purifier") for j in 1:N]
  ρ_flat  = flatten_tensorarray(ρ.X)
  v_flat = zeros(size(ρ_flat))
  
  η = 0.1
  γ = 0.9
  opt = SGD(η=η,γ=γ)
  
  for n in 1:100
    ∇ = generategradients(sites,links,kraus,χ,ξ,d)
    ∇_flat = flatten_tensorarray(∇)
    ρ_flat  = flatten_tensorarray(ρ.X)
    # algorithm
    update!(ρ,∇,opt)
    ρ′_alg_flat = flatten_tensorarray(ρ.X)
    # exact
    v_flat = γ * v_flat - η * ∇_flat 
    ρ′_flat = ρ_flat + v_flat 
    @test ρ′_flat ≈ ρ′_alg_flat rtol = 1e-4
  end

end

@testset "adagrad" begin
  N = 3
  χ = 4
  d = 2
  ψ = randomstate(N;χ=χ,σ=1.0)
  sites = siteinds(ψ) 
  links = linkinds(ψ)

  ψ_flat  = flatten_tensorarray(ψ)
  ∇²_flat = zeros(size(ψ_flat))
  η = 0.1
  ϵ = 1E-8
  opt = AdaGrad(;η=η,ϵ=ϵ)
  
  for n in 1:100
    ∇ = generategradients(sites,links,χ,d)
    ∇_flat = flatten_tensorarray(∇)
    ψ_flat  = flatten_tensorarray(ψ)
    # algorithm
    update!(ψ,∇,opt)
    ψ′_alg_flat = flatten_tensorarray(ψ)
    # exact
    ∇²_flat += ∇_flat .^ 2
    g_flat = sqrt.(∇²_flat .+ ϵ)
    ψ′_flat = ψ_flat - η * (∇_flat ./ g_flat)
    @test ψ′_flat ≈ ψ′_alg_flat rtol = 1e-4
  end
  
  N = 3
  χ = 4
  ξ = 3
  d = 2
  ρ = randomstate(N;mixed=true,χ=χ,ξ=ξ,σ=1.0)
  sites = firstsiteinds(ρ.X) 
  links = linkinds(ρ.X)
  kraus = [firstind(ρ.X[j],tags="Purifier") for j in 1:N]
  ρ_flat  = flatten_tensorarray(ρ.X)
  ∇²_flat = zeros(size(ρ_flat))
  η = 0.1
  ϵ = 1E-8

  opt = AdaGrad(;η=η,ϵ=ϵ)

  for n in 1:100
    ∇ = generategradients(sites,links,kraus,χ,ξ,d)
    ∇_flat = flatten_tensorarray(∇)
    ρ_flat  = flatten_tensorarray(ρ.X)
    # algorithm
    update!(ρ,∇,opt)
    ρ′_alg_flat = flatten_tensorarray(ρ.X)
    # exact
    ∇²_flat += ∇_flat .^ 2
    g_flat = sqrt.(∇²_flat .+ ϵ)
    ρ′_flat = ρ_flat - η * (∇_flat ./ g_flat)
    @test ρ′_flat ≈ ρ′_alg_flat rtol = 1e-4
  end
end

@testset "adadelta" begin
  N = 3
  χ = 4
  d = 2
  ψ = randomstate(N;χ=χ,σ=1.0)
  sites = siteinds(ψ) 
  links = linkinds(ψ)
  
  ψ_flat  = flatten_tensorarray(ψ)
  ∇²_flat = zeros(size(ψ_flat))
  Δθ²_flat = zeros(size(ψ_flat))
  γ = 0.9
  ϵ = 1E-8
  opt = AdaDelta(;γ=γ,ϵ=ϵ)
  
  for n in 1:100
    ∇ = generategradients(sites,links,χ,d)
    ∇_flat = flatten_tensorarray(∇)
    ψ_flat  = flatten_tensorarray(ψ)
    # algorithm
    update!(ψ,∇,opt)
    ψ′_alg_flat = flatten_tensorarray(ψ)
    # exact
    ∇²_flat = γ * ∇²_flat + (1-γ) * ∇_flat .^2
    g1_flat = sqrt.(∇²_flat .+ ϵ)
    g2_flat = sqrt.(Δθ²_flat .+ ϵ)
    Δ_flat = ∇_flat ./ g1_flat
    Δθ_flat =  Δ_flat .* g2_flat
    ψ′_flat = ψ_flat - Δθ_flat
    Δθ²_flat = γ * Δθ²_flat + (1-γ) * Δθ_flat .^2
    @test ψ′_flat ≈ ψ′_alg_flat rtol = 1e-4
  end
  N = 3
  χ = 4
  ξ = 3
  d = 2
  ρ = randomstate(N;mixed=true,χ=χ,ξ=ξ,σ=1.0)
  sites = firstsiteinds(ρ.X) 
  links = linkinds(ρ.X)
  kraus = [firstind(ρ.X[j],tags="Purifier") for j in 1:N]
  ρ_flat  = flatten_tensorarray(ρ.X)
  ∇²_flat = zeros(size(ρ_flat))
  Δθ²_flat = zeros(size(ρ_flat))
  γ = 0.9
  ϵ = 1E-8
  opt = AdaDelta(;γ=γ,ϵ=ϵ)

  for n in 1:100
    ∇ = generategradients(sites,links,kraus,χ,ξ,d)
    ∇_flat = flatten_tensorarray(∇)
    ρ_flat  = flatten_tensorarray(ρ.X)
    # algorithm
    update!(ρ,∇,opt)
    ρ′_alg_flat = flatten_tensorarray(ρ.X)
    # exact
    ∇²_flat = γ * ∇²_flat + (1-γ) * ∇_flat .^2
    g1_flat = sqrt.(∇²_flat .+ ϵ)
    g2_flat = sqrt.(Δθ²_flat .+ ϵ)
    Δ_flat = ∇_flat ./ g1_flat
    Δθ_flat =  Δ_flat .* g2_flat
    ρ′_flat = ρ_flat - Δθ_flat
    Δθ²_flat = γ * Δθ²_flat + (1-γ) * Δθ_flat .^2
    @test ρ′_flat ≈ ρ′_alg_flat rtol = 1e-4
  end
end


@testset "adam" begin
  N = 3
  χ = 4
  d = 2
  ψ = randomstate(N;χ=χ,σ=1.0)
  sites = siteinds(ψ) 
  links = linkinds(ψ)
  
  ψ_flat  = flatten_tensorarray(ψ)
  g_flat  = zeros(size(ψ_flat))
  g²_flat = zeros(size(ψ_flat))
  β₁ = 0.9
  β₂ = 0.999
  η  = 0.01
  ϵ  = 1E-8
  opt = Adam(;η=η,β₁=β₁,β₂=β₂,ϵ=ϵ)
  
  for n in 1:100
    ∇ = generategradients(sites,links,χ,d)
    ∇_flat  = flatten_tensorarray(∇)
    ψ_flat  = flatten_tensorarray(ψ)
    # algorithm
    update!(ψ,∇,opt;step=n)
    ψ′_alg_flat = flatten_tensorarray(ψ)
    # exact
    g_flat  = β₁ * g_flat  + (1-β₁) * ∇_flat
    g²_flat = β₂ * g²_flat + (1-β₂) * (∇_flat .^2)
    ĝ_flat  = g_flat  / (1-β₁^n)
    ĝ²_flat = g²_flat / (1-β₂^n)
    g1_flat = sqrt.(ĝ²_flat) .+ ϵ
    Δθ_flat = ĝ_flat ./ g1_flat
    ψ′_flat = ψ_flat - η * Δθ_flat
    @test ψ′_flat ≈ ψ′_alg_flat rtol = 1e-4
  end
  
  N = 3
  χ = 4
  ξ = 3
  d = 2
  ρ = randomstate(N;mixed=true,χ=χ,ξ=ξ,σ=1.0)
  sites = firstsiteinds(ρ.X) 
  links = linkinds(ρ.X)
  kraus = [firstind(ρ.X[j],tags="Purifier") for j in 1:N]
  ρ_flat  = flatten_tensorarray(ρ.X)
  g_flat  = zeros(size(ρ_flat))
  g²_flat = zeros(size(ρ_flat))
  β₁ = 0.9
  β₂ = 0.999
  η  = 0.01
  ϵ  = 1E-8
  opt = Adam(;η=η,β₁=β₁,β₂=β₂,ϵ=ϵ)
  
  for n in 1:100
    ∇ = generategradients(sites,links,kraus,χ,ξ,d)
    ∇_flat = flatten_tensorarray(∇)
    ρ_flat  = flatten_tensorarray(ρ.X)
    # algorithm
    update!(ρ,∇,opt;step=n)
    ρ′_alg_flat = flatten_tensorarray(ρ.X)
    # exact
    g_flat  = β₁ * g_flat  + (1-β₁) * ∇_flat
    g²_flat = β₂ * g²_flat + (1-β₂) * (∇_flat .^2)
    ĝ_flat  = g_flat  / (1-β₁^n)
    ĝ²_flat = g²_flat / (1-β₂^n)
    g1_flat = sqrt.(ĝ²_flat) .+ ϵ
    Δθ_flat = ĝ_flat ./ g1_flat
    ρ′_flat = ρ_flat - η * Δθ_flat
    @test ρ′_flat ≈ ρ′_alg_flat rtol = 1e-4
  end
end


#@testset "adamax" begin
#  N = 3
#  χ = 4
#  d = 2
#  ψ = randomstate(N,χ)
#  sites = siteinds(ψ) 
#  links = linkinds(ψ)
#
#  ψ_flat  = flatten_tensorarray(ψ)
#  g_flat  = zeros(size(ψ_flat))
#  u_flat = zeros(size(ψ_flat))
#  β₁ = 0.9
#  β₂ = 0.999
#  η  = 0.01
#  opt = AdaMax(ψ;η=η,β₁=β₁,β₂=β₂)
#  
#  for n in 1:1
#    ∇ = generategradients(sites,links,χ,d)
#    ∇_flat  = flatten_tensorarray(∇)
#    ψ_flat  = flatten_tensorarray(ψ)
#    
#    ## algorithm
#    #update!(ψ,∇,opt;step=n)
#    #ψ′_alg_flat = flatten_tensorarray(ψ)
#    
#    ## exact
#    g_flat  = β₁ * g_flat  + (1-β₁) * ∇_flat
#    u_flat  = max.(β₂ * u_flat,abs.(g_flat))
#    
#    Δθ_flat = g_flat ./ u_flat
#    
#    ψ′_flat = ψ_flat - (η/(1-β₁^n)) * Δθ_flat
#    
#    #@test ψ′_flat ≈ ψ′_alg_flat rtol = 1e-4
#
#  end
#end


#@testset "nadam" begin
#  N = 3
#  χ = 4
#  d = 2
#  ψ = randomstate(N,χ)
#  sites = siteinds(ψ) 
#  links = linkinds(ψ)
#  
#  ψ_flat  = flatten_tensorarray(ψ)
#  g_flat  = zeros(size(ψ_flat))
#  g²_flat = zeros(size(ψ_flat))
#  β₁ = 0.9
#  β₂ = 0.999
#  η  = 0.01
#  ϵ  = 1E-8
#  opt = Nadam(ψ;η=η,β₁=β₁,β₂=β₂,ϵ=ϵ)
#  
#  for n in 1:100
#    ∇ = generategradients(sites,link,χ,d)
#    ∇_flat  = flatten_tensorarray(∇)
#    ψ_flat  = flatten_tensorarray(ψ)
#    
#    # algorithm
#    update!(ψ,∇,opt;step=n)
#    ψ′_alg_flat = flatten_tensorarray(ψ)
#    
#    # exact
#    
#
#    #g_flat  = β₁ * g_flat  + (1-β₁) * ∇_flat
#    #g²_flat = β₂ * g²_flat + (1-β₂) * (∇_flat .^2)
#    #
#    #ĝ_flat  = g_flat  / (1-β₁^n)
#    #ĝ²_flat = g²_flat / (1-β₂^n)
#    #
#    #g1_flat = sqrt.(ĝ²_flat) .+ ϵ
#    #Δθ_flat = ĝ_flat ./ g1_flat
#    #
#    #ψ′_flat = ψ_flat - η * Δθ_flat
#    
#    @test ψ′_flat ≈ ψ′_alg_flat rtol = 1e-4
#
#  end
#end


