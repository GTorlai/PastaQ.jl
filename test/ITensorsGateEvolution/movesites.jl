using Test
using PastaQ
using .ITensorsGateEvolution
using ITensors
using Combinatorics

@testset "movesites" begin

  @testset "$N sites" for N in 1:7
    s0 = siteinds("qubit", N)
    ψ0 = productMPS(s0, "0")
    for perm in permutations(1:N)
      s = s0[perm]
      ψ = productMPS(s, rand(("0", "1"), N))
      ns = 1:N
      ns′ = [findsite(ψ0, i) for i in s]
      @test ns′ == perm
      ψ′ = movesites(ψ, ns .=> ns′)
      for n in 1:N
        @test siteind(ψ0, n) == siteind(ψ′, n)
      end
      @test prod(ψ) ≈ prod(ψ′)
    end
  end

end

