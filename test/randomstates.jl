using PastaQ
using ITensors
using Test
using LinearAlgebra
using Random

@testset "randomstate: MPS" begin
  N = 5
  χ = 3
  
  # Real-valued with randpars
  ψ = randomstate(Float64, N; χ = χ)
  @test length(ψ) == N
  @test maxlinkdim(ψ) == χ 
  for j in 1:length(ψ)
    @test eltype(ψ[j]) == Float64
  end
  # Real-valued with circuit
  ψ = randomstate(Float64, N; χ = χ, alg = "circuit")
  @test length(ψ) == N
  @test maxlinkdim(ψ) == χ 
  for j in 1:length(ψ)
    @test eltype(ψ[j]) == Float64
  end
  # Complex-valued with randpars
  ψ = randomstate(N; χ = χ)
  @test length(ψ) == N
  @test maxlinkdim(ψ) == χ 
  for j in 1:length(ψ)
    @test eltype(ψ[j]) == Complex{Float64}
  end
  
  # TODO: complex-valued with circuit: not implemented

  # Normalization
  ψ = randomstate(Float64, N; χ = χ, normalize = true)
  @test norm(ψ)^2 ≈ 1
  ψ = randomstate(Float64, N; χ = χ, alg = "circuit",
                  normalize = true)
  @test norm(ψ)^2 ≈ 1
  ψ = randomstate(N; χ = χ, normalize = true)
  @test norm(ψ)^2 ≈ 1
  
end

@testset "randomstate: LPDO" begin
  N = 5
  χ = 3
  ξ = 2
  
  # Real-valued with randpars
  ρ = randomstate(Float64, N; χ = χ, ξ = ξ, mixed = true)
  @test length(ρ) == N
  @test maxlinkdim(ρ.X) == χ 
  for j in 1:length(ρ)
    @test eltype(ρ.X[j]) == Float64
  end
  ρ_mat = array(ρ)
  @test sum(abs.(imag(diag(ρ_mat)))) ≈ 0.0 atol=1e-10
  @test all(real(eigvals(ρ_mat)) .≥ 0) 
  # Complex-valued with randpars
  ρ = randomstate(N; χ = χ, ξ = ξ, mixed = true)
  @test length(ρ) == N
  @test maxlinkdim(ρ.X) == χ 
  for j in 1:length(ρ)
    @test eltype(ρ.X[j]) == Complex{Float64}
  end
  ρ_mat = array(ρ)
  @test sum(abs.(imag(diag(ρ_mat)))) ≈ 0.0 atol=1e-10
  @test all(real(eigvals(ρ_mat)) .≥ 0) 
  
  # TODO: randomLPDO with circuit + thermal state not implemented

  # Normalization
  ρ = randomstate(Float64, N; χ = χ, normalize = true, mixed = true)
  @test tr(ρ) ≈ 1
  ρ = randomstate(N; χ = χ, normalize = true, mixed = true)
  @test tr(ρ) ≈ 1
  
end


@testset "randomprocess: MPO" begin
  N = 5
  χ = 3
  
  # Real-valued with randpars
  U = randomprocess(Float64, N; χ = χ, mixed = false)
  @test length(U) == N
  @test maxlinkdim(U) == χ 
  for j in 1:length(U)
    @test eltype(U[j]) == Float64
  end
  # Complex-valued with randpars
  U = randomprocess(N; χ = χ, mixed = false)
  @test length(U) == N
  @test maxlinkdim(U) == χ 
  for j in 1:length(U)
    @test eltype(U[j]) == Complex{Float64}
  end
  
end


@testset "randomprocess: LPDO" begin
  N = 4
  χ = 3
  ξ = 2
  
  # Real-valued with randpars
  Λ = randomprocess(Float64, N; χ = χ, ξ = ξ, mixed = true)
  @test length(Λ) == N
  @test maxlinkdim(Λ.X) == χ 
  for j in 1:length(Λ)
    @test eltype(Λ.X[j]) == Float64
  end
  # Complex-valued with randpars
  Λ = randomprocess(N; χ = χ, ξ = ξ, mixed = true)
  @test length(Λ) == N
  @test maxlinkdim(Λ.X) == χ 
  for j in 1:length(Λ)
    @test eltype(Λ.X[j]) == Complex{Float64}
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
  U0 = randomprocess(N;mixed=false,χ=χ)

  # to MPO
  U = randomprocess(U0;mixed=false)
  for j in 1:length(N)
    @test firstind(U[j],tags="Site",plev=0) == firstind(U0[j],tags="Site",plev=0)
  end
  # to Choi
  Λ = randomprocess(U0;mixed=true)
  Γ0 = LPDO(PastaQ._unitaryMPO_to_choiMPS(U0))
  for j in 1:length(N)
    @test firstind(Λ.X[j],tags="Input") == firstind(Γ0.X[j],tags="Input")
    @test firstind(Λ.X[j],tags="Output") == firstind(Γ0.X[j],tags="Output")
  end
  
  # 1. Given a Choi
  Λ0 = randomprocess(N;mixed=true,χ=χ)
  # to MPO
  U = randomprocess(Λ0;mixed=false)
  for j in 1:length(N)
    @test firstind(U[j],tags="Site",plev=0).id == firstind(Λ0.X[j],tags="Output").id
    @test firstind(U[j],tags="Site",plev=1).id == firstind(Λ0.X[j],tags="Input").id
  end
  
  # to Choi
  Λ = randomprocess(Λ0;mixed=true)
  for j in 1:length(N)
    @test firstind(Λ.X[j],tags="Input") == firstind(Λ0.X[j],tags="Input")
    @test firstind(Λ.X[j],tags="Output") == firstind(Λ0.X[j],tags="Output")
  end
  
end
