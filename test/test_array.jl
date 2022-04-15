using PastaQ
using ITensors
using Test
using LinearAlgebra

@testset "array" begin
  N = 5
  ψ = productstate(N)
  ψvec = PastaQ.array(ψ)
  @test size(ψvec) == (1 << N,)

  ψ = productstate(N)
  ρ = outer(ψ', ψ)
  ρmat = PastaQ.array(ρ)
  @test size(ρmat) == (1 << N, 1 << N)

  ρ = randomstate(N; mixed=true)
  ρmat = PastaQ.array(ρ)
  @test size(ρmat) == (1 << N, 1 << N)

  U = randomprocess(N)
  Umat = PastaQ.array(U)
  @test size(Umat) == (1 << N, 1 << N)

  N = 3
  Λ = randomprocess(N; mixed=true)
end

@testset "array for MPS/MPO" begin
  N = 5
  depth = 3
  circuit = randomcircuit(N; depth=depth)
  qubits = siteinds("Qubit", N)

  # MPS wavefunction
  ψ = runcircuit(qubits, circuit)
  ψvec = PastaQ.array(ψ)

  basis = vec(reverse.(Iterators.product(fill([0, 1], N)...)))

  for (k, s) in enumerate(basis)
    σ = productstate(qubits, [s...])
    ψσ = inner(σ, ψ)
    @test ψσ ≈ ψvec[k]
  end

  # MPO unitary
  U = runcircuit(qubits, circuit; process=true)
  Umat = PastaQ.array(U)

  basis = vec(reverse.(Iterators.product(fill([0, 1], N)...)))

  for (k, s) in enumerate(basis)
    σ = productstate(qubits, [s...])
    for (kp, sp) in enumerate(basis)
      σp = productstate(qubits, [sp...])
      Uσσp = inner(σ', U, σp)
      @test Uσσp ≈ Umat[k, kp]
    end
  end

  # MPO density matrix
  ρ = runcircuit(qubits, circuit; noise=("DEP", (p=0.01,)))
  ρmat = PastaQ.array(ρ)

  basis = vec(reverse.(Iterators.product(fill([0, 1], N)...)))

  for (k, s) in enumerate(basis)
    σ = productstate(qubits, [s...])
    for (kp, sp) in enumerate(basis)
      σp = productstate(qubits, [sp...])
      ρσσp = inner(σ', ρ, σp)
      @test ρσσp ≈ ρmat[k, kp]
    end
  end

  # LPDO density matrix
  ρ = PastaQ.normalize!(randomstate(qubits; χ=4, ξ=2))
  ρmat = PastaQ.array(ρ)

  ρmpo = MPO(ρ)
  basis = vec(reverse.(Iterators.product(fill([0, 1], N)...)))

  for (k, s) in enumerate(basis)
    σ = productstate(qubits, [s...])
    for (kp, sp) in enumerate(basis)
      σp = productstate(qubits, [sp...])
      ρσσp = inner(σ', ρmpo, σp)
      @test ρσσp ≈ ρmat[k, kp]
    end
  end

  # MPO Choi matrix
  N = 3
  depth = 3
  circuit = randomcircuit(N; depth=depth)
  qubits = siteinds("Qubit", N)

  Λ = runcircuit(qubits, circuit; process=true, noise=("DEP", (p=0.01,)))
  Λmat = PastaQ.array(Λ)

  basis = vec(reverse.(Iterators.product(fill([0, 1], 2 * N)...)))
  qubits_in = [firstind(Λ[j]; tags="Input", plev=0) for j in 1:N]
  qubits_out = [firstind(Λ[j]; tags="Output", plev=0) for j in 1:N]

  for (k, s) in enumerate(basis)
    s_in = s[1:2:end]
    s_out = s[2:2:end]
    for (kp, sp) in enumerate(basis)
      sp_in = sp[1:2:end]
      sp_out = sp[2:2:end]
      Λc = copy(Λ)
      for j in 1:N
        Λc[j] =
          Λc[j] *
          prime(state(qubits_in[j], s_in[j] + 1)) *
          state(qubits_in[j], sp_in[j] + 1)
      end
      σ = productstate(qubits_out, [s_out...])
      σp = productstate(qubits_out, [sp_out...])
      Λel = inner(σ', Λc, σp)
      @test Λel ≈ Λmat[k, kp]
    end
  end

  # LPDO Choi matrix

  Λ = PastaQ.normalize!(randomprocess(qubits; χ=4, ξ=2); localnorm=2)
  Λmat = PastaQ.array(Λ)

  Λmpo = MPO(Λ)
  qubits_in = [firstind(Λ.X[j]; tags="Input", plev=0) for j in 1:N]
  qubits_out = [firstind(Λ.X[j]; tags="Output", plev=0) for j in 1:N]

  for (k, s) in enumerate(basis)
    s_in = s[1:2:end]
    s_out = s[2:2:end]
    for (kp, sp) in enumerate(basis)
      sp_in = sp[1:2:end]
      sp_out = sp[2:2:end]
      Λc = copy(Λmpo)
      for j in 1:N
        Λc[j] =
          Λc[j] *
          prime(state(qubits_in[j], s_in[j] + 1)) *
          state(qubits_in[j], sp_in[j] + 1)
      end
      σ = productstate(qubits_out, [s_out...])
      σp = productstate(qubits_out, [sp_out...])
      Λel = inner(σ', Λc, σp)
      @test Λel ≈ Λmat[k, kp]
    end
  end
end

@testset "array for full representation" begin
  N = 5
  depth = 3
  circuit = randomcircuit(N; depth=depth)
  qubits = siteinds("Qubit", N)

  # MPS wavefunction
  ψ = runcircuit(qubits, circuit)
  ψvec = PastaQ.array(ψ)
  ψprod = prod(ψ)
  ψtest = PastaQ.array(ψprod)
  @test ψvec ≈ ψtest

  # MPO unitary
  U = runcircuit(qubits, circuit; process=true)
  Umat = PastaQ.array(U)
  Uprod = prod(U)
  Utest = PastaQ.array(Uprod)
  @test Umat ≈ Utest

  # MPO density matrix
  ρ = runcircuit(qubits, circuit; noise=("DEP", (p=0.01,)))
  ρmat = PastaQ.array(ρ)
  ρprod = prod(ρ)
  ρtest = PastaQ.array(ρprod)
  @test ρtest ≈ ρmat

  # LPDO density matrix
  ρ = PastaQ.normalize!(randomstate(qubits; χ=4, ξ=2))
  ρmat = PastaQ.array(ρ)
  ρprod = prod(ρ)
  ρtest = PastaQ.array(ρprod)
  @test ρtest ≈ ρmat

  # MPO Choi matrix
  N = 3
  depth = 3
  circuit = randomcircuit(N; depth=depth)
  qubits = siteinds("Qubit", N)

  Λ = runcircuit(qubits, circuit; process=true, noise=("DEP", (p=0.01,)))
  Λmat = PastaQ.array(Λ)
  Λprod = prod(Λ)
  Λtest = PastaQ.array(Λprod)
  @test Λmat ≈ Λtest

  # LPDO Choi matrix
  Λ = PastaQ.normalize!(randomprocess(qubits; χ=4, ξ=2); localnorm=2)
  Λmat = PastaQ.array(Λ)
  Λprod = prod(Λ)
  Λtest = PastaQ.array(Λprod)
  @test Λmat ≈ Λtest
end

@testset "dense states to itensors" begin
  N = 2
  d = 1 << N
  gates = randomcircuit(N; depth=4)
  ψ = runcircuit(N, gates)

  sites = siteinds("Qubit", N)
  ψvec = PastaQ.array(ψ)
  ϕ = PastaQ.itensor(ψvec, sites)
  @test PastaQ.array(ϕ) ≈ ψvec

  ρ = runcircuit(N, gates; noise=("DEP", (p=0.1,)))
  ρmat = PastaQ.array(ρ)
  ϱ = PastaQ.itensor(ρmat, sites)
  @test PastaQ.array(ϱ) ≈ ρmat
end
