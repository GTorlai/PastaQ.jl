xusing PastaQ
using LinearAlgebra
using Test

@testset "linear array" begin
  N = 10
  couplings = PastaQ.lineararray(N)
  @test length(couplings) == 2
  @test length(couplings[1]) == N ÷ 2
  @test length(couplings[2]) == (N ÷ 2 - 1)

  for cycle in couplings
    q = collect(1:N)
    for bond in cycle
      @test bond[1] in q
      deleteat!(q, findfirst(x -> x == bond[1], q))
      @test bond[2] in q
      deleteat!(q, findfirst(x -> x == bond[2], q))
    end
  end

  N = 11
  couplings = PastaQ.lineararray(N)
  @test length(couplings) == 2
  @test length(couplings[1]) == N ÷ 2
  @test length(couplings[2]) == N ÷ 2
  for cycle in couplings
    q = collect(1:N)
    for bond in cycle
      @test bond[1] in q
      deleteat!(q, findfirst(x -> x == bond[1], q))
      @test bond[2] in q
      deleteat!(q, findfirst(x -> x == bond[2], q))
    end
  end
end

@testset "square array" begin
  Lx = 4
  Ly = 4

  couplings = PastaQ.squarearray(Lx, Ly)
  @test length(couplings) == 4
  @test length(couplings[1]) == Lx * (Ly ÷ 2)
  @test length(couplings[2]) == (Lx ÷ 2) * (Ly ÷ 2)
  @test length(couplings[3]) == Lx * (Ly ÷ 2)
  @test length(couplings[4]) == (Lx ÷ 2) * (Ly ÷ 2)

  N = Lx * Ly
  for cycle in couplings
    q = collect(1:N)
    for bond in cycle
      @test bond[1] in q
      deleteat!(q, findfirst(x -> x == bond[1], q))
      @test bond[2] in q
      deleteat!(q, findfirst(x -> x == bond[2], q))
    end
  end

  couplings = PastaQ.squarearray(Lx, Ly; rotated=true)
  for cycle in couplings
    q = collect(1:N)
    for bond in cycle
      @test bond[1] in q
      deleteat!(q, findfirst(x -> x == bond[1], q))
      @test bond[2] in q
      deleteat!(q, findfirst(x -> x == bond[2], q))
    end
  end

  Lx = 5
  Ly = 5

  couplings = PastaQ.squarearray(Lx, Ly)
  @test length(couplings) == 4
  @test length(couplings[1]) == Lx * (Ly ÷ 2)
  @test length(couplings[2]) == (Lx ÷ 2) * Ly
  @test length(couplings[3]) == Lx * (Ly ÷ 2)
  @test length(couplings[4]) == Lx * (Ly ÷ 2)

  N = Lx * Ly
  for cycle in couplings
    q = collect(1:N)
    for bond in cycle
      @test bond[1] in q
      deleteat!(q, findfirst(x -> x == bond[1], q))
      @test bond[2] in q
      deleteat!(q, findfirst(x -> x == bond[2], q))
    end
  end

  couplings = PastaQ.squarearray(Lx, Ly; rotated=true)
  for cycle in couplings
    q = collect(1:N)
    for bond in cycle
      @test bond[1] in q
      deleteat!(q, findfirst(x -> x == bond[1], q))
      @test bond[2] in q
      deleteat!(q, findfirst(x -> x == bond[2], q))
    end
  end
end

@testset "random couplints" begin
  N = 20
  R = 10

  for d in 1:10
    q = collect(1:N)
    bonds = PastaQ.randomcouplings(N, R)
    for bond in bonds
      @test bond[1] in q
      deleteat!(q, findfirst(x -> x == bond[1], q))
      @test bond[2] in q
      deleteat!(q, findfirst(x -> x == bond[2], q))
    end
  end
end
