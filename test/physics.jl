using PastaQ
using LinearAlgebra
using ITensors
using Test

function exact_densitymatrix(N,β,h)
  id_mat = [1 0;0 1]
  Z = [1 0;0 -1]
  ZZ = kron(Z,Z)
  X = [0 1;1 0]
  H = zeros(1<<N,1<<N)

  for j in 1:N-1
    x = 1.0
    for k in 1:N-1
      if k == j
        x = kron(x,ZZ)
      else
        x = kron(x,id_mat)
      end
    end
    H = H - x
  end
  for j in 1:N
    x = 1.0
    for k in 1:N
      if k == j
        x = kron(x,X)
      else
        x = kron(x,id_mat)
      end
    end
    H = H - h * x
  end
  rho = exp(-β * H)
  rho = rho / tr(rho)
  return rho
end

@testset " transverse field ising model " begin
  N = 5
  h = 1.0
  ψ = transversefieldising(N,h)
  @test length(ψ) == N
  N = 5
  β = 0.5
  τ = 0.001
  ρ = transversefieldising(N,h,β=β,τ=τ)
  @test length(ρ) == N
  ρ_mat = fullmatrix(ρ)
  @test ρ_mat ≈ transpose(conj(ρ_mat)) atol=1e-2
  exact_ρ = exact_densitymatrix(N,β,h)
  @test ρ_mat ≈ exact_ρ atol=1e-2
end

