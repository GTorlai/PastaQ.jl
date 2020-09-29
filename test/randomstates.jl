using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random

@testset "randomstate: MPS" begin
  N = 5
  χ = 3
  
  # Real-valued with randpars
  ψ = randomstate(N;χ=χ,complex=false)
  @test length(ψ) == N
  @test maxlinkdim(ψ) == χ 
  for j in 1:length(ψ)
    @test eltype(ψ[j]) == Float64
  end
  # Real-valued with circuit
  ψ = randomstate(N;χ=χ,init="circuit",complex=false)
  @test length(ψ) == N
  @test maxlinkdim(ψ) == χ 
  for j in 1:length(ψ)
    @test eltype(ψ[j]) == Float64
  end
  # Complex-valued with randpars
  ψ = randomstate(N;χ=χ,complex=true)
  @test length(ψ) == N
  @test maxlinkdim(ψ) == χ 
  for j in 1:length(ψ)
    @test eltype(ψ[j]) == Complex{Float64}
  end
  
  # TODO: complex-valued with circuit: not implemented

  # Normalization
  ψ = randomstate(N;χ=χ,normalize=true,complex=false)
  @test norm(ψ)^2 ≈ 1
  ψ = randomstate(N;χ=χ,init="circuit",normalize=true,complex=false)
  @test norm(ψ)^2 ≈ 1
  ψ = randomstate(N;χ=χ,complex=true,normalize=true)
  @test norm(ψ)^2 ≈ 1
  
end

@testset "randomstate: LPDO" begin
  N = 5
  χ = 3
  ξ = 2
  
  # Real-valued with randpars
  ρ = randomstate(N;χ=χ,ξ=ξ,mixed=true,complex=false)
  @test length(ρ) == N
  @test maxlinkdim(ρ.X) == χ 
  for j in 1:length(ρ)
    @test eltype(ρ.X[j]) == Float64
  end
  ρ_mat = fullmatrix(ρ)
  @test sum(abs.(imag(diag(ρ_mat)))) ≈ 0.0 atol=1e-10
  @test all(real(eigvals(ρ_mat)) .≥ 0) 
  # Complex-valued with randpars
  ρ = randomstate(N;χ=χ,ξ=ξ,mixed=true,complex=true)
  @test length(ρ) == N
  @test maxlinkdim(ρ.X) == χ 
  for j in 1:length(ρ)
    @test eltype(ρ.X[j]) == Complex{Float64}
  end
  ρ_mat = fullmatrix(ρ)
  @test sum(abs.(imag(diag(ρ_mat)))) ≈ 0.0 atol=1e-10
  @test all(real(eigvals(ρ_mat)) .≥ 0) 
  
  # TODO: randomLPDO with circuit + thermal state not implemented

  # Normalization
  ρ = randomstate(N;χ=χ,normalize=true,mixed=true,complex=false)
  @test tr(ρ) ≈ 1
  ρ = randomstate(N;χ=χ,complex=true,normalize=true,mixed=true)
  @test tr(ρ) ≈ 1
  
end


@testset "randomprocess: MPO" begin
  N = 5
  χ = 3
  
  # Real-valued with randpars
  ρ = randomprocess(N;χ=χ,mixed=false,complex=false)
  @test length(ρ) == N
  @test maxlinkdim(ρ) == χ 
  for j in 1:length(ρ)
    @test eltype(ρ[j]) == Float64
  end
  # Complex-valued with randpars
  ρ = randomstate(N;χ=χ,mixed=false,complex=true)
  @test length(ρ) == N
  @test maxlinkdim(ρ) == χ 
  for j in 1:length(ρ)
    @test eltype(ρ[j]) == Complex{Float64}
  end
  
end


@testset "randomprocess: LPDO" begin
  N = 4
  χ = 3
  ξ = 2
  
  # Real-valued with randpars
  Λ = randomprocess(N;χ=χ,ξ=ξ,mixed=true,complex=false)
  ρ = Λ.M
  @test length(ρ) == N
  @test maxlinkdim(ρ.X) == χ 
  for j in 1:length(ρ)
    @test eltype(ρ.X[j]) == Float64
  end
  # Complex-valued with randpars
  Λ = randomprocess(N;χ=χ,ξ=ξ,mixed=true,complex=true)
  ρ = Λ.M
  @test length(ρ) == N
  @test maxlinkdim(ρ.X) == χ 
  for j in 1:length(ρ)
    @test eltype(ρ.X[j]) == Complex{Float64}
  end
end



@testset "initialization given a state" begin
  # Complex-valued with randpars
  N = 3
  χ = 3

  Ψ0 = randomstate(N;χ=χ)
  Ψ  = randomstate(Ψ0)  
  for j in 1:length(N)
    @test firstind(Ψ[j],tags="Site") == firstind(Ψ0[j],tags="Site")
  end
  
  ρ = randomstate(Ψ0;mixed=true)
  for j in 1:length(N)
    @test firstind(ρ.X[j],tags="Site",plev=0) == firstind(Ψ0[j],tags="Site")
  end

  ## 1. Given a LPDO
  ρ0 = randomstate(N;mixed=true,χ=χ)

  # to MPS
  Ψ = randomstate(ρ0)
  for j in 1:length(N)
    @test firstind(Ψ[j],tags="Site") == firstind(ρ0.X[j],tags="Site",plev=0)
  end
  # to LPDO
  ρ = randomstate(ρ0;mixed=true)
  for j in 1:length(N)
    @test firstind(ρ.X[j],tags="Site",plev=0) == firstind(ρ0.X[j],tags="Site",plev=0)
  end

end



@testset "initialization given a process" begin
  # Complex-valued with randpars
  N = 3
  χ = 3

  # 1. Given a MPO
  ρ0 = randomprocess(N;mixed=false,χ=χ)

  ## to MPO
  ρ = randomprocess(ρ0;mixed=false)
  for j in 1:length(N)
    @test firstind(ρ[j],tags="Site",plev=0) == firstind(ρ0[j],tags="Site",plev=0)
  end

  Λ = randomprocess(ρ0;mixed=true)
  ρ = Λ.M
  for j in 1:length(N)
    @test firstind(ρ.X[j],tags="Site",plev=0) == firstind(ρ0[j],tags="Site",plev=0)
  end

  ## 1. Given a LPDO
  Λ0 = randomprocess(N;mixed=true,χ=χ)
  ρ0 = Λ0.M 

  ## to MPO
  ρ = randomprocess(ρ0;mixed=false)
  for j in 1:length(N)
    @test firstind(ρ[j],tags="Site",plev=0) == firstind(ρ0.X[j],tags="Site",plev=0)
  end
  ## to LPDO
  Λ = randomprocess(ρ0;mixed=true)
  ρ = Λ.M
  for j in 1:length(N)
    @test firstind(ρ.X[j],tags="Site",plev=0) == firstind(ρ0.X[j],tags="Site",plev=0)
  end

end
