using PastaQ
using ITensors
using Test
using LinearAlgebra

@testset "Gate generation: 1-qubit gates" begin
  i = Index(2; tags="Qubit")

  g = gate("I", i)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  g = gate("X", i)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  @test g ≈ gate("σx", i)
  @test g ≈ gate("σ1", i)

  g = gate("Y", i)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  @test g ≈ gate("σy", i)
  @test g ≈ gate("σ2", i)

  g = gate("Z", i)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  @test g ≈ gate("σz", i)
  @test g ≈ gate("σ3", i)

  g = gate("√X", i)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  @test g ≈ gate("√NOT", i)

  g = gate("H", i)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  g = gate("S", i)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  @test g ≈ gate("P", i)

  g = gate("T", i)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  g = state("X+", i)
  @test plev(ind(g, 1)) == 0

  g = state("X-", i)
  @test plev(ind(g, 1)) == 0

  g = state("Y+", i)
  @test plev(ind(g, 1)) == 0

  g = state("Y-", i)
  @test plev(ind(g, 1)) == 0

  g = state("Z+", i)
  @test plev(ind(g, 1)) == 0

  g = state("Z-", i)
  @test plev(ind(g, 1)) == 0

  θ = π * rand()
  g = gate("Rx", i; θ=θ)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  θ = π * rand()
  g = gate("Ry", i; θ=θ)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  ϕ = 2π * rand()
  g = gate("Rz", i; ϕ=ϕ)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)

  angles = rand(3)
  θ = π * angles[1]
  ϕ = 2π * angles[2]
  λ = 2π * angles[3]
  g = gate("Rn", i; θ=θ, ϕ=ϕ, λ=λ)
  @test plev(inds(g)[1]) == 1
  @test plev(inds(g)[2]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ Matrix{Int}(I, 2, 2)
  @test g ≈ gate("Rn̂", i; θ=θ, ϕ=ϕ, λ=λ)
end

@testset "Gate generation: 2-qubit gates" begin
  i = Index(2)
  j = Index(2)
  angles = rand(3)
  θ = π * angles[1]
  ϕ = 2π * angles[2]
  λ = 2π * angles[3]

  g = gate("SWAP", i, j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4), (2, 2, 2, 2))
  @test g ≈ gate("Swap", i, j)
  @test g ≈ gate("Sw", i, j)

  g = gate("√SWAP", i, j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4), (2, 2, 2, 2))
  @test g ≈ gate("√Swap", i, j)
  @test g ≈ gate("√Sw", i, j)

  g = gate("iSWAP", i, j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test g ≈ gate("iSwap", i, j)
  @test g ≈ gate("iSw", i, j)

  g = gate("CX", i, j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4), (2, 2, 2, 2))

  g = gate("CY", i, j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4), (2, 2, 2, 2))

  g = gate("CZ", i, j)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4), (2, 2, 2, 2))

  g = gate("CRz", i, j; ϕ=ϕ)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4), (2, 2, 2, 2))

  g = gate("CRn", i, j; θ=θ, ϕ=ϕ, λ=λ)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1
  @test plev(inds(g)[3]) == 0 && plev(inds(g)[4]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 4, 4), (2, 2, 2, 2))
  @test g ≈ gate("CRn̂", i, j; θ=θ, ϕ=ϕ, λ=λ)
end

@testset "Gate generation: 3-qubit gates" begin
  i = Index(2)
  j = Index(2)
  k = Index(2)

  g = gate("Toffoli", i, j, k)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1 && plev(inds(g)[3]) == 1
  @test plev(inds(g)[4]) == 0 && plev(inds(g)[5]) == 0 && plev(inds(g)[6]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 8, 8), (2, 2, 2, 2, 2, 2))
  @test g ≈ gate("CCNOT", i, j, k)
  @test g ≈ gate("CCX", i, j, k)
  @test g ≈ gate("TOFF", i, j, k)

  g = gate("Fredkin", i, j, k)
  @test plev(inds(g)[1]) == 1 && plev(inds(g)[2]) == 1 && plev(inds(g)[3]) == 1
  @test plev(inds(g)[4]) == 0 && plev(inds(g)[5]) == 0 && plev(inds(g)[6]) == 0
  ggdag = g * prime(dag(g), 1; plev=1)
  @test ITensors.array(ggdag) ≈ reshape(Matrix{ComplexF64}(I, 8, 8), (2, 2, 2, 2, 2, 2))
  @test g ≈ gate("CSWAP", i, j, k)
  @test g ≈ gate("CSwap", i, j, k)
  @test g ≈ gate("CSw", i, j, k)
end

@testset "Gate generation: Haar gates" begin
  i = Index(2)
  j = Index(2)

  g = gate("randU", i)
  @test hasinds(g, i', i)
  Id = g * swapprime(dag(g)', 2 => 1)
  for n in 1:dim(i), n′ in 1:dim(i)
    if n == n′
      @test Id[n, n′] ≈ 1
    else
      @test Id[n, n′] ≈ 0 atol = 1e-15
    end
  end

  g = gate("randU", i, j)
  @test hasinds(g, i', j', i, j)
  Id = g * swapprime(dag(g)', 2 => 1)
  for n in 1:dim(i), m in 1:dim(j), n′ in 1:dim(i), m′ in 1:dim(j)
    if (n, m) == (n′, m′)
      @test Id[n, m, n′, m′] ≈ 1
    else
      @test Id[n, m, n′, m′] ≈ 0 atol = 1e-14
    end
  end

  @testset "Custom gate with long name" begin
    PastaQ.gate(::GateName"my_favorite_gate") = [0.11 0.12; 0.21 0.22]
    s = Index(2, "Qubit, Site")
    gate("my_favorite_gate", s)
    g = gate("my_favorite_gate", s)
    @test g[s' => 1, s => 1] == 0.11
    @test g[s' => 1, s => 2] == 0.12
    @test g[s' => 2, s => 1] == 0.21
    @test g[s' => 2, s => 2] == 0.22
  end
end

@testset "Bases rotations" begin
  zp = [1.0, 0.0]
  zm = [0.0, 1.0]
  xp = [1.0, 1.0] / √2
  xm = [1.0, -1.0] / √2
  yp = [1.0, im] / √2
  ym = [1.0, -im] / √2

  Ux = gate("basisX"; dag=true)
  @test Ux * xp ≈ zp
  @test Ux * xm ≈ zm

  Ux = gate("basisY"; dag=true)
  @test Ux * yp ≈ zp
  @test Ux * ym ≈ zm
end

@testset "apply gate: Id" begin
  psi = productstate(2)
  gate_data = ("I", 1)
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi) ≈ [1.0, 0.0, 0.0, 0.0]
end

@testset "apply gate: X" begin
  # Build gate first, then apply using an ITensor
  psi = productstate(2)
  site = 1
  gate_data = ("X", 1)
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi) ≈ [0.0, 0.0, 1.0, 0.0]
end

@testset "apply gate: Y" begin
  psi = productstate(2)
  site = 1
  gate_data = ("Y", 1)
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi) ≈ [0.0, 0.0, im, 0.0]
end

@testset "apply gate: Z" begin
  # Build gate first, then apply using an ITensor
  psi = productstate(2)
  site = 1
  gate_data = ("Z", 1)
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi) ≈ [1.0, 0.0, 0.0, 0.0]
end

@testset "apply gate: H" begin
  # Build gate first, then apply using an ITensor
  psi = productstate(2)
  site = 1
  gate_data = ("H", 1)
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi) ≈ 1 / sqrt(2.0) * [1.0, 0.0, 1.0, 0.0]
end

@testset "apply gate: S" begin
  psi = productstate(1)
  gate_data = ("S", 1)
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi[1]) ≈ [1.0, 0.0]
  psi = productstate(1)
  gate_data = ("X", 1)
  psi = runcircuit(psi, gate_data)
  gate_data = ("S", 1)
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi[1]) ≈ [0.0, im]
end

@testset "apply gate: T" begin
  psi = productstate(1)
  gate_data = ("T", 1)
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi[1]) ≈ [1.0, 0.0]
  psi = productstate(1)
  gate_data = ("X", 1)
  psi = runcircuit(psi, gate_data)
  gate_data = ("T", 1)
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi[1]) ≈ [0.0, exp(im * π / 4)]
end

@testset "apply gate: Rx" begin
  θ = π * rand()
  psi = productstate(1)
  gate_data = ("Rx", 1, (θ=θ,))
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi[1]) ≈ [cos(θ / 2.0), -im * sin(θ / 2.0)]
  psi = productstate(1)
  psi = runcircuit(psi, ("X", 1))
  #PastaQ.applygate!(psi,"X",1)
  gate_data = ("Rx", 1, (θ=θ,))
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi[1]) ≈ [-im * sin(θ / 2.0), cos(θ / 2.0)]
end

@testset "apply gate: Ry" begin
  θ = π * rand()
  psi = productstate(1)
  gate_data = ("Ry", 1, (θ=θ,))
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi[1]) ≈ [cos(θ / 2.0), sin(θ / 2.0)]
  psi = productstate(1)
  psi = runcircuit(psi, ("X", 1))
  gate_data = ("Ry", 1, (θ=θ,))
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi[1]) ≈ [-sin(θ / 2.0), cos(θ / 2.0)]
end

@testset "apply gate: Rz" begin
  ϕ = 2π * rand()
  psi = productstate(1)
  gate_data = ("Rz", 1, (ϕ=ϕ,))
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi[1]) ≈ [1.0, 0.0]
  psi = productstate(1)
  psi = runcircuit(psi, ("X", 1))
  gate_data = ("Rz", 1, (ϕ=ϕ,))
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi[1]) ≈ [0.0, exp(im * ϕ)]
end

@testset "apply gate: Rn" begin
  θ = 1.0
  ϕ = 2.0
  λ = 3.0
  psi = productstate(1)
  gate_data = ("Rn", 1, (θ=θ, ϕ=ϕ, λ=λ))
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi[1]) ≈ [cos(θ / 2.0), exp(im * ϕ) * sin(θ / 2.0)]
  psi = productstate(1)
  psi = runcircuit(psi, ("X", 1))
  gate_data = ("Rn", 1, (θ=θ, ϕ=ϕ, λ=λ))
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi[1]) ≈
    [-exp(im * λ) * sin(θ / 2.0), exp(im * (ϕ + λ)) * cos(θ / 2.0)]
end

@testset "apply gate: prep X+/X-" begin
  psi = productstate(1, ["X+"])
  @test PastaQ.array(psi) ≈ 1 / sqrt(2.0) * [1.0, 1.0]

  psi = productstate(1, ["X-"])
  @test PastaQ.array(psi) ≈ 1 / sqrt(2.0) * [1.0, -1.0]
end

@testset "apply gate: prep Y+/Y-" begin
  psi = productstate(1, ["Y+"])
  @test PastaQ.array(psi) ≈ 1 / sqrt(2.0) * [1.0, im]

  psi = productstate(1, ["Y-"])
  @test PastaQ.array(psi) ≈ 1 / sqrt(2.0) * [1.0, -im]
end

@testset "apply gate: prep Z+/Z-" begin
  psi = productstate(1, ["Z+"])
  @test PastaQ.array(psi) ≈ [1.0, 0.0]

  psi = productstate(1, ["Z-"])
  @test PastaQ.array(psi) ≈ [0.0, 1.0]
end

@testset "apply gate: meas X" begin
  psi = productstate(1, ["X+"])
  gate_data = ("basisX", 1, (dag=true,))
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi) ≈ [1.0, 0.0]
  psi = productstate(1, ["X-"])
  gate_data = ("basisX", 1, (dag=true,))
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi) ≈ [0.0, 1.0]
end

@testset "apply gate: meas Y" begin
  psi = productstate(1, ["Y+"])
  gate_data = ("basisY", 1, (dag=true,))
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi) ≈ [1.0, 0.0]
  psi = productstate(1, ["Y-"])
  gate_data = ("basisY", 1, (dag=true,))
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi) ≈ [0.0, 1.0]
end

@testset "apply gate: meas Z" begin
  psi = productstate(1)
  gate_data = ("basisZ", 1, (dag=true,))
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi) ≈ [1.0, 0.0]
  psi = productstate(1, ["Z-"])
  gate_data = ("basisZ", 1, (dag=true,))
  psi = runcircuit(psi, gate_data)
  @test PastaQ.array(psi) ≈ [0.0, 1.0]
end

@testset "apply gate: CX" begin
  # CONTROL - TARGET
  psi = productstate(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  gate_data = ("CX", (1, 2))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [1.0, 0.0, 0.0, 0.0]

  psi = productstate(2)
  # |10> -> |11> = (0 0 0 1) (natural order)
  gate_data = ("X", 1)
  psi = runcircuit(psi, gate_data)
  gate_data = ("CX", (1, 2))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, 0.0, 0.0, 1.0]

  psi = productstate(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  gate_data = ("X", 2)
  psi = runcircuit(psi, gate_data)
  gate_data = ("CX", (1, 2))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, 1.0, 0.0, 0.0]

  psi = productstate(2)
  # |11> -> |10> = (0 0 1 0) (natural order)
  gate_data = ("X", 1)
  psi = runcircuit(psi, gate_data)
  gate_data = ("X", 2)
  psi = runcircuit(psi, gate_data)
  gate_data = ("CX", (1, 2))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, 0.0, 1.0, 0.0]
end

@testset "apply gate: CY" begin
  # CONTROL - TARGET
  psi = productstate(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  gate_data = ("CY", (1, 2))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [1.0, 0.0, 0.0, 0.0]

  psi = productstate(2)
  # |10> -> i|11> = (0 0 0 i) (natural order)
  gate_data = ("X", 1)
  psi = runcircuit(psi, gate_data)
  gate_data = ("CY", (1, 2))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, 0.0, 0.0, im]

  psi = productstate(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  gate_data = ("X", 2)
  psi = runcircuit(psi, gate_data)
  gate_data = ("CY", (1, 2))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, 1.0, 0.0, 0.0]

  psi = productstate(2)
  # |11> -> -i|10> = (0 0 -i 0) (natural order)
  gate_data = ("X", 1)
  psi = runcircuit(psi, gate_data)
  gate_data = ("X", 2)
  psi = runcircuit(psi, gate_data)
  gate_data = ("CY", (1, 2))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, 0.0, -im, 0.0]

  # TARGET - CONTROL
  psi = productstate(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  gate_data = ("CY", (2, 1))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [1.0, 0.0, 0.0, 0.0]

  psi = productstate(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  gate_data = ("X", 1)
  psi = runcircuit(psi, gate_data)
  gate_data = ("CY", (2, 1))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, 0.0, 1.0, 0.0]

  psi = productstate(2)
  # |01> -> i|11> = (0 0 0 i) (natural order)
  gate_data = ("X", 2)
  psi = runcircuit(psi, gate_data)
  gate_data = ("CY", (2, 1))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, 0.0, 0.0, im]

  psi = productstate(2)
  # |11> -> -i|01> = (0 -i 0 0) (natural order)
  gate_data = ("X", 1)
  psi = runcircuit(psi, gate_data)
  gate_data = ("X", 2)
  psi = runcircuit(psi, gate_data)
  gate_data = ("CY", (2, 1))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, -im, 0.0, 0.0]
end

@testset "apply gate: CZ" begin
  # CONTROL - TARGET
  psi = productstate(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  gate_data = ("CZ", (1, 2))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [1.0, 0.0, 0.0, 0.0]

  psi = productstate(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  gate_data = ("X", 1)
  psi = runcircuit(psi, gate_data)
  gate_data = ("CZ", (1, 2))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, 0.0, 1.0, 0.0]

  psi = productstate(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  gate_data = ("X", 2)
  psi = runcircuit(psi, gate_data)
  gate_data = ("CZ", (1, 2))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, 1.0, 0.0, 0.0]

  psi = productstate(2)
  # |11> -> -|11> = (0 0 0 -1) (natural order)
  gate_data = ("X", 1)
  psi = runcircuit(psi, gate_data)
  gate_data = ("X", 2)
  psi = runcircuit(psi, gate_data)
  gate_data = ("CZ", (1, 2))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, 0.0, 0.0, -1.0]

  # CONTROL - TARGET
  psi = productstate(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  gate_data = ("CZ", (2, 1))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [1.0, 0.0, 0.0, 0.0]

  psi = productstate(2)
  # |10> -> |10> = (0 0 1 0) (natural order)
  gate_data = ("X", 1)
  psi = runcircuit(psi, gate_data)
  gate_data = ("CZ", (2, 1))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, 0.0, 1.0, 0.0]

  psi = productstate(2)
  # |01> -> |01> = (0 1 0 0) (natural order)
  gate_data = ("X", 2)
  psi = runcircuit(psi, gate_data)
  gate_data = ("CZ", (2, 1))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, 1.0, 0.0, 0.0]

  psi = productstate(2)
  # |11> -> -|11> = (0 0 0 -1) (natural order)
  gate_data = ("X", 1)
  psi = runcircuit(psi, gate_data)
  gate_data = ("X", 2)
  psi = runcircuit(psi, gate_data)
  gate_data = ("CZ", (2, 1))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, 0.0, 0.0, -1.0]
end

@testset "apply gate: Sw" begin
  # CONTROL - TARGET
  psi = productstate(2)
  # |00> -> |00> = (1 0 0 0) (natural order)
  gate_data = ("Sw", (1, 2))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [1.0, 0.0, 0.0, 0.0]

  psi = productstate(2)
  # |10> -> |01> = (0 1 0 0) (natural order)
  gate_data = ("X", 1)
  psi = runcircuit(psi, gate_data)
  gate_data = ("Sw", (1, 2))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, 1.0, 0.0, 0.0]

  psi = productstate(2)
  # |01> -> |10> = (0 0 1 0) (natural order)
  gate_data = ("X", 2)
  psi = runcircuit(psi, gate_data)
  gate_data = ("Sw", (1, 2))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, 0.0, 1.0, 0.0]

  psi = productstate(2)
  # |11> -> |11> = (0 0 0 1) (natural order)
  gate_data = ("X", 1)
  psi = runcircuit(psi, gate_data)
  gate_data = ("X", 2)
  psi = runcircuit(psi, gate_data)
  gate_data = ("Sw", (1, 2))
  psi = runcircuit(psi, gate_data)
  psi_vec = PastaQ.array(psi)
  @test psi_vec ≈ [0.0, 0.0, 0.0, 1.0]
end

@testset "function applied to a gate" begin
  s = siteinds("Qubit", 2)

  θ = 0.1
  rx = gate("Rx"; θ=0.1)
  exp_rx = exp(rx)
  gtest = gate("Rx", s[1]; θ=0.1, f=x -> exp(x))
  @test PastaQ.array(gtest) ≈ exp_rx

  cx = 0.1 * gate("CX")
  exp_cx = exp(cx)
  gtest = gate("CX", s[1], s[2]; f=x -> exp(0.1 * x))
  @test PastaQ.array(gtest) ≈ exp_cx
end

@testset "gate built out of elementary gates" begin
  s = siteinds("Qubit", 2)

  G = gate("S") * gate("S")
  gtest = gate("S * S", s[1])
  @test PastaQ.array(gtest) ≈ G

  cx = gate("CX")

  G = cx * cx * cx
  gtest = gate("CX*CX*CX", s[1], s[2])
  @test PastaQ.array(gtest) ≈ G

  G = gate("S") + gate("T")
  gtest = gate("S + T", s[1])
  @test PastaQ.array(gtest) ≈ G

  G = gate("S") + gate("T") * gate("X")
  gtest = gate("S + T * X", s[1])
  @test PastaQ.array(gtest) ≈ G

  G = gate("S") + gate("T") * gate("X") * gate("Y")
  gtest = gate("S + T * X * Y", s[1])
  @test PastaQ.array(gtest) ≈ G

  G = gate("Z") * gate("S") + gate("T") * gate("X") * gate("Y")
  gtest = gate("Z*S + T * X * Y", s[1])
  @test PastaQ.array(gtest) ≈ G

  exp_cx = exp(0.1 * cx)
  gtest = gate("CX", s[1], s[2]; f=x -> exp(0.1 * x))
  @test PastaQ.array(gtest) ≈ exp_cx

  d = 3
  q = siteinds("Qudit", 4; dim=d)
  g1 = gate("a†a", q[1]; f=x -> exp(0.1 * x))
  g2 = gate("a† * a", q[1]; f=x -> exp(0.1 * x))
  @test g1 ≈ g2
end

@testset "qudit gates" begin
  dim = 3
  s = siteinds("Qudit", 4; dim=dim)
  dims = (dim,)
  @test gate("a†", dims) ≈ [0 0 0; 1 0 0; 0 √2 0]
  @test gate("a", dims) ≈ [0 1 0; 0 0 √2; 0 0 0]

  @test gate("a†", dims) * gate("a", dims) ≈ [0 0 0; 0 1 0; 0 0 2]

  @test PastaQ.array(gate("a†", s[1])) ≈ [0 0 0; 1 0 0; 0 √2 0]
  @test PastaQ.array(gate("a", s[1])) ≈ [0 1 0; 0 0 √2; 0 0 0]

  @test PastaQ.array(gate("a†a", s[1])) ≈ [0 0 0; 0 1 0; 0 0 2]
  @test PastaQ.array(gate("aa†", s[1])) ≈ [1 0 0; 0 2 0; 0 0 0]
  @test PastaQ.array(gate("a† * a", s[1])) ≈ [0 0 0; 0 1 0; 0 0 2]
  @test PastaQ.array(gate("a * a†", s[1])) ≈ [1 0 0; 0 2 0; 0 0 0]

  dim = 10
  s = siteinds("Qudit", 4; dim=dim)
  dims = (dim,)
  @test PastaQ.array(gate("a† * a† * a * a", s[1])) ≈
    gate("a†", dims) * gate("a†", dims) * gate("a", dims) * gate("a", dims)

  @test PastaQ.array(gate("a†a", s[1], s[2])) ≈ kron(gate("a†", dims), gate("a", dims))
  @test PastaQ.array(gate("aa†", s[1], s[2])) ≈ kron(gate("a", dims), gate("a†", dims))

  @test PastaQ.array(gate("a†a+aa†", s[1], s[2])) ≈
    kron(gate("a†", dims), gate("a", dims)) + kron(gate("a", dims), gate("a†", dims))
end
