using PastaQ
using Random
using ITensors
using Test
using LinearAlgebra

#
# Helper functions for tests
#

function state_to_int(state::Array)
  index = 0
  for j in 1:length(state)
    index += 2^(j - 1) * state[length(state) + 1 - j]
  end
  return index
end

function empiricalprobability(samples::Matrix)
  prob = zeros((1 << size(samples)[2]))
  for n in 1:size(samples)[1]
    sample = samples[n, :]
    index = state_to_int(sample)
    prob[index + 1] += 1
  end
  prob = prob / size(samples)[1]
  return prob
end

@testset "generation of preparation states" begin
  N = 4
  nstates = 100
  states = PastaQ.randompreparations(N, nstates)
  @test size(states)[1] == nstates
  @test size(states)[2] == N
end

@testset "generation of measurement bases" begin
  N = 4
  nbases = 100
  bases = randombases(N, nbases)
  @test size(bases)[1] == nbases
  @test size(bases)[2] == N

  bases = fullbases(N)
  @test bases isa Matrix{String}
  @test size(bases) == (3^N,N)
end

@testset "measurements" begin
  Random.seed!(1234)
  N = 3
  depth = 4
  ψ0 = productstate(N)
  gates = randomcircuit(N; depth =  depth)
  ψ = runcircuit(ψ0, gates)
  ψ_vec = PastaQ.array(ψ)
  probs = abs2.(ψ_vec)

  nshots = 50000
  samples = PastaQ.getsamples(ψ, nshots)
  @test size(samples)[1] == nshots
  @test size(samples)[2] == N
  data_prob = empiricalprobability(samples)
  @test probs ≈ data_prob atol = 1e-2
  
  ρ = runcircuit(N, gates; noise=("amplitude_damping", (γ=0.01,)))
  ρ_mat = PastaQ.array(ρ)
  probs = real(diag(ρ_mat))

  samples = PastaQ.getsamples(ρ, nshots)
  @test size(samples)[1] == nshots
  @test size(samples)[2] == N
  data_prob = empiricalprobability(samples)
  @test probs ≈ data_prob atol = 1e-2
end

@testset "basis rotations" begin
  s = siteinds("Qubit",1)
  #ϕ = qubits(1)
  ψ0 = state("X+",s[1])
  gates = [("basisX",1,(dag=true,))]
  ψ = runcircuit(ψ0,gates)
  @test PastaQ.array(ψ) ≈ state("Z+")
  ψ0 = state("X-",s[1])
  gates = [("basisX",1,(dag=true,))]
  ψ = runcircuit(ψ0,gates)
  @test PastaQ.array(ψ) ≈ state("Z-")

  ψ0 = state("Y+",s[1])
  gates = [("basisY",1,(dag=true,))]
  ψ = runcircuit(ψ0,gates)
  @test PastaQ.array(ψ) ≈ state("Z+")
  ψ0 = state("Y-",s[1])
  gates = [("basisY",1,(dag=true,))]
  ψ = runcircuit(ψ0,gates)
  @test PastaQ.array(ψ) ≈ state("Z-")
end


@testset "project unitary MPO" begin
  N = 4
  ntrial = 100
  gates = randomcircuit(N; depth =  4, layered=false)

  U = runcircuit(N, gates; process=true)

  bases = randombases(N, ntrial)
  preps = randompreparations(N, ntrial)

  for n in 1:ntrial
    mgates = PastaQ.measurementgates(bases[n, :])
    ψ_in = productstate(N, preps[n, :])
    ψ_out = runcircuit(ψ_in, gates)

    Ψ_out = PastaQ.projectchannel(U, preps[n, :])
    @test PastaQ.array(ψ_out) ≈ PastaQ.array(Ψ_out)

    ψ_m = runcircuit(ψ_out, mgates)
    Ψ_m = runcircuit(Ψ_out, mgates)
    @test PastaQ.array(ψ_m) ≈ PastaQ.array(Ψ_m)
  end
end

@testset "project unitary ITensor" begin
  N = 4
  ntrial = 100
  gates = randomcircuit(N; depth =  4, layered=false)

  U = runcircuit(N, gates; process=true, full_representation = true)
  bases = randombases(N, ntrial)
  preps = randompreparations(N, ntrial)

  for n in 1:ntrial
    mgates = PastaQ.measurementgates(bases[n, :])
    ψ_in = productstate(N, preps[n, :])
    ψ_out = runcircuit(ψ_in, gates)

    Ψ_out = PastaQ.projectchannel(U, preps[n, :])
    @test PastaQ.array(ψ_out) ≈ PastaQ.array(Ψ_out)

    ψ_m = runcircuit(ψ_out, mgates)
    Ψ_m = runcircuit(Ψ_out, mgates)
    @test PastaQ.array(ψ_m) ≈ PastaQ.array(Ψ_m)
  end
end

@testset "project Choi MPO" begin
  N = 4
  ntrial = 100
  gates = randomcircuit(N; depth =  4, layered=false)

  Λ = runcircuit(N, gates; process=true, noise=("amplitude_damping", (γ=0.1,)))

  bases = randombases(N, ntrial)
  preps = PastaQ.randompreparations(N, ntrial)
  for n in 1:ntrial
    mgates = PastaQ.measurementgates(bases[n, :])
    ψ_in = productstate(N, preps[n, :])
    ρ_out = runcircuit(ψ_in, gates; noise=("amplitude_damping", (γ=0.1,)))

    Λ_out = PastaQ.projectchannel(Λ, preps[n, :])
    @test PastaQ.array(ρ_out) ≈ PastaQ.array(Λ_out) atol = 1e-6

    ρ_m = runcircuit(ρ_out, mgates)
    Λ_m = runcircuit(Λ_out, mgates)
    @test PastaQ.array(ρ_m) ≈ PastaQ.array(Λ_m) atol = 1e-6
  end
end

@testset "project Choi ITensor" begin
  N = 4
  ntrial = 100
  gates = randomcircuit(N; depth =  4, layered=false)

  @disable_warn_order begin
    Λ = runcircuit(N, gates; process=true, noise=("amplitude_damping", (γ=0.1,)), full_representation = true)

    bases = randombases(N, ntrial)
    preps = PastaQ.randompreparations(N, ntrial)
    for n in 1:ntrial
      mgates = PastaQ.measurementgates(bases[n, :])
      ψ_in = productstate(N, preps[n, :])
      ρ_out = runcircuit(ψ_in, gates; noise=("amplitude_damping", (γ=0.1,)))

      Λ_out = PastaQ.projectchannel(Λ, preps[n, :])
      @test PastaQ.array(ρ_out) ≈ PastaQ.array(Λ_out) atol = 1e-6

      ρ_m = runcircuit(ρ_out, mgates)
      Λ_m = runcircuit(Λ_out, mgates)
      @test PastaQ.array(ρ_m) ≈ PastaQ.array(Λ_m) atol = 1e-6
    end
  end
end






@testset "getsamples: states" begin
  N = 3
  circuit = randomcircuit(N; depth = 3)
  
  # quantum states
  ψ = runcircuit(N, circuit)
  ρ = runcircuit(N, circuit; noise=("amplitude_damping", (γ=0.1,)))
  
  nbases = 11
  bases = randombases(N, nbases)
  data = getsamples(ψ, bases)
  @test size(data) == (nbases,N)
  nshots = 3
  data = getsamples(ψ, bases, nshots)
  @test size(data) == (nbases*nshots,N)
  for b in 1:nbases
    for k in 1:nshots
      @test first.(data[(b-1)*nshots+k,:]) == bases[b,:]
    end
  end

  data = getsamples(ρ, bases)
  @test size(data) == (nbases,N)
  nshots = 3
  data = getsamples(ρ, bases, nshots)
  @test size(data) == (nbases*nshots,N)
  for b in 1:nbases
    for k in 1:nshots
      @test first.(data[(b-1)*nshots+k,:]) == bases[b,:]
    end
  end
  
  # ITensors quantum states
  ψ = runcircuit(N, circuit; full_representation = true)
  ρ = runcircuit(N, circuit; noise=("amplitude_damping", (γ=0.1,)), full_representation = true)
  
  nbases = 11
  bases = randombases(N, nbases)
  data = getsamples(ψ, bases)
  @test size(data) == (nbases,N)
  nshots = 3
  data = getsamples(ψ, bases, nshots)
  @test size(data) == (nbases*nshots,N)
  for b in 1:nbases
    for k in 1:nshots
      @test first.(data[(b-1)*nshots+k,:]) == bases[b,:]
    end
  end

  data = getsamples(ρ, bases)
  @test size(data) == (nbases,N)
  nshots = 3
  data = getsamples(ρ, bases, nshots)
  @test size(data) == (nbases*nshots,N)
  for b in 1:nbases
    for k in 1:nshots
      @test first.(data[(b-1)*nshots+k,:]) == bases[b,:]
    end
  end
end

@testset "getsamples: channels" begin

  N = 3
  circuit = randomcircuit(N; depth = 3)
  
  # quantum processes
  U = runcircuit(N, circuit; process = true)
  Λ = runcircuit(N, circuit; noise=("amplitude_damping", (γ=0.1,)), process = true)
  
  npreps = 5
  nbases = 7
  preps = randompreparations(N, npreps)
  bases = randombases(N, nbases)
  
  data = getsamples(U, preps, bases)
  @test size(data) == (npreps*nbases,N)
  nshots = 3
  data = getsamples(U, preps, bases, nshots)
  @test size(data) == (npreps*nbases*nshots,N)
  
  for p in 1:npreps
    for b in 1:nbases
      for k in 1:nshots
        pdata = first.(data[(p-1)*nbases*nshots+(b-1)*nshots+k,:])
        bdata = first.(last.(data[(p-1)*nbases*nshots+(b-1)*nshots+k,:]))
        @test pdata == preps[p,:]
        @test bdata == bases[b,:]
      end
    end
  end
  
  npreps = 5
  nbases = 5
  preps = randompreparations(N, npreps)
  bases = randombases(N, nbases)
  
  data = getsamples(U, preps .=> bases)
  @test size(data) == (npreps,N)


  data = getsamples(Λ, preps, bases)
  @test size(data) == (npreps*nbases,N)
  nshots = 3
  data = getsamples(Λ, preps, bases, nshots)
  @test size(data) == (npreps*nbases*nshots,N)
  
  for p in 1:npreps
    for b in 1:nbases
      for k in 1:nshots
        pdata = first.(data[(p-1)*nbases*nshots+(b-1)*nshots+k,:])
        bdata = first.(last.(data[(p-1)*nbases*nshots+(b-1)*nshots+k,:]))
        @test pdata == preps[p,:]
        @test bdata == bases[b,:]
      end
    end
  end
  
  npreps = 5
  nbases = 5
  preps = randompreparations(N, npreps)
  bases = randombases(N, nbases)
  
  data = getsamples(Λ, preps .=> bases)
  @test size(data) == (npreps,N)

  # full representation
  U = runcircuit(N, circuit; process = true, full_representation = true)
  Λ = runcircuit(N, circuit; noise=("amplitude_damping", (γ=0.1,)), process = true,full_representation = true)
  
  npreps = 5
  nbases = 7
  preps = randompreparations(N, npreps)
  bases = randombases(N, nbases)
  
  data = getsamples(U, preps, bases)
  @test size(data) == (npreps*nbases,N)
  nshots = 3
  data = getsamples(U, preps, bases, nshots)
  @test size(data) == (npreps*nbases*nshots,N)
  
  for p in 1:npreps
    for b in 1:nbases
      for k in 1:nshots
        pdata = first.(data[(p-1)*nbases*nshots+(b-1)*nshots+k,:])
        bdata = first.(last.(data[(p-1)*nbases*nshots+(b-1)*nshots+k,:]))
        @test pdata == preps[p,:]
        @test bdata == bases[b,:]
      end
    end
  end
  
  data = getsamples(Λ, preps, bases)
  @test size(data) == (npreps*nbases,N)
  nshots = 3
  data = getsamples(Λ, preps, bases, nshots)
  @test size(data) == (npreps*nbases*nshots,N)
  
  for p in 1:npreps
    for b in 1:nbases
      for k in 1:nshots
        pdata = first.(data[(p-1)*nbases*nshots+(b-1)*nshots+k,:])
        bdata = first.(last.(data[(p-1)*nbases*nshots+(b-1)*nshots+k,:]))
        @test pdata == preps[p,:]
        @test bdata == bases[b,:]
      end
    end
  end
end

