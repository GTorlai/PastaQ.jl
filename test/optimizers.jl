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
  

@testset "sgd" begin
  N = 3
  χ = 4
  d = 2
  ψ = initializetomography(N;χ=χ,σ=1.0)
  sites = siteinds(ψ) 
  links = linkinds(ψ)

  ψ_flat  = flatten_tensorarray(ψ)
  v_flat = zeros(size(ψ_flat))
  
  η = 0.1
  γ = 0.9
  opt = SGD(ψ;η=η,γ=γ)
  
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

end

@testset "adagrad" begin
  N = 3
  χ = 4
  d = 2
  ψ = initializetomography(N;χ=χ)
  sites = siteinds(ψ) 
  links = linkinds(ψ)

  ψ_flat  = flatten_tensorarray(ψ)
  ∇²_flat = zeros(size(ψ_flat))
  η = 0.1
  ϵ = 1E-8
  opt = AdaGrad(ψ;η=η,ϵ=ϵ)
  
  for n in 1:2
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
end

@testset "adadelta" begin
  N = 3
  χ = 4
  d = 2
  ψ = initializetomography(N;χ=χ)
  sites = siteinds(ψ) 
  links = linkinds(ψ)
  
  ψ_flat  = flatten_tensorarray(ψ)
  ∇²_flat = zeros(size(ψ_flat))
  Δθ²_flat = zeros(size(ψ_flat))
  γ = 0.9
  ϵ = 1E-8
  opt = AdaDelta(ψ;γ=γ,ϵ=ϵ)
  
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
end


@testset "adam" begin
  N = 3
  χ = 4
  d = 2
  ψ = initializetomography(N;χ=χ)
  sites = siteinds(ψ) 
  links = linkinds(ψ)
  
  ψ_flat  = flatten_tensorarray(ψ)
  g_flat  = zeros(size(ψ_flat))
  g²_flat = zeros(size(ψ_flat))
  β₁ = 0.9
  β₂ = 0.999
  η  = 0.01
  ϵ  = 1E-8
  opt = Adam(ψ;η=η,β₁=β₁,β₂=β₂,ϵ=ϵ)
  
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
end


#@testset "adamax" begin
#  N = 3
#  χ = 4
#  d = 2
#  ψ = initializetomography(N,χ)
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
#  ψ = initializetomography(N,χ)
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


