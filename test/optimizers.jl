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

@testset "sgd" begin
  N = 3
  χ = 4
  d = 2
  ψ = initializetomography(N,χ;σ=1.0)
  sites = siteinds(ψ) 
  links = linkinds(ψ)

  σ = 1e-1
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
     
  ψ_flat = flatten_tensorarray(ψ)
  ∇_flat = flatten_tensorarray(∇)

  η = 0.1
  opt = SGD(ψ;η=η)
  
  # Exact
  ψ′_flat = ψ_flat - η * ∇_flat
  
  # First gradient update
  update!(ψ,∇,opt)
  ∇ψ_flat = flatten_tensorarray(ψ)
  
  @test ψ′_flat ≈ ∇ψ_flat

end

@testset "adagrad" begin
  N = 3
  χ = 4
  d = 2
  ψ = initializetomography(N,χ)
  sites = siteinds(ψ) 
  links = linkinds(ψ)

  function generategradients(sites,χ,d)
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
  
  ψ_flat  = flatten_tensorarray(ψ)
  ∇²_flat = zeros(size(ψ_flat))
  η = 0.1
  ϵ = 1E-8
  opt = Adagrad(ψ;η=η,ϵ=ϵ)
  
  for n in 1:2
    ∇ = generategradients(sites,χ,d)
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
  ψ = initializetomography(N,χ)
  sites = siteinds(ψ) 
  links = linkinds(ψ)

  function generategradients(sites,χ,d)
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
  
  ψ_flat  = flatten_tensorarray(ψ)
  ∇²_flat = zeros(size(ψ_flat))
  Δθ²_flat = zeros(size(ψ_flat))
  γ = 0.9
  ϵ = 1E-8
  opt = Adadelta(ψ;γ=γ,ϵ=ϵ)
  
  for n in 1:100
    ∇ = generategradients(sites,χ,d)
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


