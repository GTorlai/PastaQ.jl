using PastaQ
using ITensors
using Test
using Random

function flatten_tensorarray(T)
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

#@testset "adagrad" begin
#  N = 3
#  χ = 4
#  d = 2
#  ψ = initializetomography(N,χ)
#  sites = siteinds(ψ) 
#  links = linkinds(ψ)
#
#  σ = 1e-1
#  ∇₁ = ITensor[]
#  ∇₂ = ITensor[]
#  
#  rand_mat = σ * (ones(d,χ) - 2*rand(d,χ))
#  rand_mat += im * σ * (ones(d,χ) - 2*rand(d,χ))
#  push!(∇₁,ITensor(rand_mat,sites[1],links[1]))
#  rand_mat = σ * (ones(d,χ) - 2*rand(d,χ))
#  rand_mat += im * σ * (ones(d,χ) - 2*rand(d,χ))
#  push!(∇₂,ITensor(rand_mat,sites[1],links[1]))
#  for j in 2:N-1
#    rand_mat = σ * (ones(χ,d,χ) - 2*rand(χ,d,χ))
#    rand_mat += im * σ * (ones(χ,d,χ) - 2*rand(χ,d,χ))
#    push!(∇₁,ITensor(rand_mat,links[j-1],sites[j],links[j]))
#    rand_mat = σ * (ones(χ,d,χ) - 2*rand(χ,d,χ))
#    rand_mat += im * σ * (ones(χ,d,χ) - 2*rand(χ,d,χ))
#    push!(∇₂,ITensor(rand_mat,links[j-1],sites[j],links[j]))
#  end
#  # Site N
#  rand_mat = σ * (ones(χ,d) - 2*rand(χ,d))
#  rand_mat += im * σ * (ones(χ,d) - 2*rand(χ,d))
#  push!(∇₁,ITensor(rand_mat,links[N-1],sites[N]))
#  rand_mat = σ * (ones(χ,d) - 2*rand(χ,d))
#  rand_mat += im * σ * (ones(χ,d) - 2*rand(χ,d))
#  push!(∇₂,ITensor(rand_mat,links[N-1],sites[N]))
#     
#  
#  ψ_flat  = flatten_tensorarray(ψ)
#  ∇₁_flat = flatten_tensorarray(∇₁)
#  ∇₂_flat = flatten_tensorarray(∇₂)
#  
#  η = 0.1
#  ϵ = 1E-8
#  opt = Adagrad(ψ;η=η,ϵ=ϵ)
#
#  ## First gradient update
#  ## Exact
#  #∇₁²_flat = ∇₁_flat .^ 2  
#  #∇₁²_flat .+= ϵ
#  #for j in 1:length(∇₁²_flat)
#  #  @test ∇₁²_flat[j] ≈ ∇₁_flat[j]^2 atol=1e-5
#  #end
#  #
#  #δ₁_flat = sqrt.(∇₁²_flat) 
#  #for j in 1:length(∇₁²_flat)
#  #  @test δ₁_flat[j] ≈ sqrt(∇₁²_flat[j]) atol=1e-5
#  #end
#  #
#  #ψ′_flat = ψ_flat - η * (∇₁_flat ./ δ₁_flat)
#  #for j in 1:length(∇₁²_flat)
#  #  @test ψ′_flat[j]≈(ψ_flat[j]-η*(∇₁_flat[j]/δ₁_flat[j])) atol=1e-5
#  #end
#
#  # First gradient update
#  update!(ψ,∇₁,opt)
#  #ψ′_alg_flat = flatten_tensorarray(ψ)
#  
#  #@test ψ′_flat ≈ ψ′_alg_flat atol=1e-8
# 
##  ϕ  = fullvector(ψ)
##  ∂₁ = fullvector(MPS(∇₁))
##  ∂₂ = fullvector(MPS(∇₂))
##  
##  # First gradient update
##  update!(ψ,∇₁,opt)
##  ψ₁ = fullvector(copy(ψ))
##  
##
##
##  ## Second gradient update
##  #update!(ψ,∇₂,opt)
##  #ψ₂ = fullvector(copy(ψ))
##  ## Third gradient update
##  #update!(ψ,∇₁,opt)
##  #ψ₃ = fullvector(copy(ψ))
##
##
#end


