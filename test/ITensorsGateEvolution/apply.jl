using Test
using PastaQ
using .ITensorsGateEvolution
using ITensors

@testset "apply" begin

  @testset "Simple on-site state evolution" begin
    N = 3

    pos = ProductOps()
    pos *= "Z", 3
    pos *= "Y", 2
    pos *= "X", 1

    s = siteinds("qubit", N)
    gates = ops(s, pos)
    ψ0 = productMPS(s, "0")

    # Apply the gates
    ψ = apply(gates, ψ0)

    # Move site 1 to position 3
    ψ′ = movesite(ψ, 1, 3)
    @test siteind(ψ′, 1) == s[2]
    @test siteind(ψ′, 2) == s[3]
    @test siteind(ψ′, 3) == s[1]
    @test prod(ψ) ≈ prod(ψ′)

    # Move the site back
    ψ′′ = movesite(ψ′, 3, 1)
    @test siteind(ψ′′, 1) == s[1]
    @test siteind(ψ′′, 2) == s[2]
    @test siteind(ψ′′, 3) == s[3]
    @test prod(ψ) ≈ prod(ψ′′)
  end

  @testset "More complex evolution" begin
    N = 7

    osX = ProductOps()
    for n in 1:N
      osX *= "X", n
    end

    osZ = ProductOps()
    for n in 1:N
      osZ *= "Z", n
    end

    osRand = ProductOps()
    for n in 1:N
      osRand *= "rand", n
    end

    osSw = ProductOps()
    for n in 1:N-2
      osSw *= "Sw", n, n+2
    end

    osCx = ProductOps()
    for n in 1:N-3
      osCx *= "Cx", n, n+3
    end

    osT = ProductOps()
    for n in 1:N-3
      osT *= "T", n, n+1, n+3
    end

    osRx = ProductOps()
    for n in 1:N
      osRx *= "Rx", n, (θ = π,)
    end

    os_noise = ProductOps()
    for n in 1:N-4
      os_noise *= "noise", n, n+2, n+4
    end

    os = osRand * osX * osSw * osRx * osZ * osCx * osT
    #os = osX * osSw * osRx * osZ * osCx * osT
    s = siteinds("qubit", N)
    gates = ops(s, os)

    @testset "Pure state evolution" begin
      ψ0 = productMPS(s, "0")
      maxdim = prod(dim(siteinds(ψ0, j)) for j in 1:N÷2)
      ψ = apply(gates, ψ0; cutoff = 1e-15, maxdim = maxdim)
      @test maxlinkdim(ψ) == maxdim
      prodψ = apply(gates, prod(ψ0))
      @test prod(ψ) ≈ prodψ
    end

    M0 = MPO(s, "Id")
    maxdim = prod(dim(siteinds(M0, j)) for j in 1:N÷2)

    @testset "Mixed state evolution" begin
      M = apply(gates, M0; cutoff = 1e-15, maxdim = maxdim)
      @test maxlinkdim(M) == 30
      sM0 = siteinds(M0)
      sM = siteinds(M)
      for n in 1:N
        @test hassameinds(sM[n], sM0[n])
      end
      set_warn_itensor_order!(15)
      prodM = apply(gates, prod(M0))
      @test prod(M) ≈ prodM
      reset_warn_itensor_order!()
    end

    @testset "Mixed state noisy evolution" begin
      os *= os_noise
      gates = ops(s, os)
      M = apply(gates, M0; apply_dag = true,
                           cutoff = 1e-15, maxdim = maxdim)
      @test maxlinkdim(M) == maxdim
      sM0 = siteinds(M0)
      sM = siteinds(M)
      for n in 1:N
        @test hassameinds(sM[n], sM0[n])
      end
      set_warn_itensor_order!(16)
      prodM = apply(gates, prod(M0); apply_dag = true)
      @test prod(M) ≈ prodM
      reset_warn_itensor_order!()
    end

    @testset "Mixed state noisy evolution" begin
      os *= os_noise
      gates = ops(s, os)
      M = apply(gates, M0; apply_dag = true,
                           cutoff = 1e-15, maxdim = maxdim-1)
      @test maxlinkdim(M) == maxdim-1
      sM0 = siteinds(M0)
      sM = siteinds(M)
      for n in 1:N
        @test hassameinds(sM[n], sM0[n])
      end
      set_warn_itensor_order!(16)
      prodM = apply(gates, prod(M0); apply_dag = true)
      @test prod(M) ≈ prodM rtol = 1e-1
      reset_warn_itensor_order!()
    end

  end

  @testset "movesites" begin
    N = 6
    s = siteinds("qubit", N)
    ψ0 = randomMPS(s)
    ψ = ITensorsGateEvolution.movesites(ψ0, 1:N, reverse(1:N))
    for n in 1:N
      @test siteind(ψ, n) == s[N-n+1]
    end
  end

end
