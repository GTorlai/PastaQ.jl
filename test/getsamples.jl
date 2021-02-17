using PastaQ
 
using ITensors
using Test
using LinearAlgebra

#
# Helper functions for tests
#

function state_to_int(state::Array)
  index = 0
  for j in 1:length(state)
    index += 2^(j-1)*state[length(state)+1-j]
  end
  return index
end

function empiricalprobability(samples::Matrix)
  prob = zeros((1<<size(samples)[2]))
  for n in 1:size(samples)[1]
    sample = samples[n,:]
    index = state_to_int(sample)
    prob[index+1] += 1
  end
  prob = prob / size(samples)[1]
  return prob
end

@testset "generation of preparation states" begin
  N = 4
  nshots = 100
  states = PastaQ.randompreparations(N,nshots)
  @test size(states)[1] == nshots
  @test size(states)[2] == N
  
  states = PastaQ.randompreparations(N, nshots, ndistinctstates = 10)
  @test size(states)[1] == nshots
  @test size(states)[2] == N
  
  for i in 1:10
    for j in 1:10
      @test states[10*(i-1)+j] == states[10*(i-1)+1]
    end
  end
end

@testset "generation of measurement bases" begin
  N = 4
  nshots = 100
  bases = randombases(N, nshots)
  @test size(bases)[1] == nshots
  @test size(bases)[2] == N
  
  bases = randombases(N, nshots, ndistinctbases = 10)
  @test size(bases)[1] == nshots
  @test size(bases)[2] == N
  
  for i in 1:10
    for j in 1:10
      @test bases[10*(i-1)+j] == bases[10*(i-1)+1]
    end
  end
end


@testset "measurements" begin
  N = 4
  depth = 10
  ψ0 = productstate(N)
  gates = randomcircuit(N,depth)
  ψ = runcircuit(ψ0,gates)
  ψ_vec = PastaQ.array(ψ)
  probs = abs2.(ψ_vec)
  
  nshots = 100000
  samples = PastaQ.getsamples!(ψ,nshots)
  @test size(samples)[1] == nshots
  @test size(samples)[2] == N
  data_prob = empiricalprobability(samples)
  @test probs ≈ data_prob atol=1e-2

  ρ = runcircuit(N, gates, noise = ("amplitude_damping", (γ = 0.01,)))
  ρ_mat = PastaQ.array(ρ)
  probs = real(diag(ρ_mat))

  samples = PastaQ.getsamples!(ρ,nshots)
  @test size(samples)[1] == nshots
  @test size(samples)[2] == N
  data_prob = empiricalprobability(samples)
  @test probs ≈ data_prob atol=1e-2
end


@testset "measurement projections" begin
  N = 8
  nshots = 20
  ψ0 = productstate(N)
  bases = randombases(N,nshots)
  
  depth = 8
  gates = randomcircuit(N,depth)
  ψ = runcircuit(ψ0,gates)
  s = siteinds(ψ)

  for n in 1:nshots
    basis = bases[n,:]
    meas_gates = PastaQ.measurementgates(basis)
    #meas_tensors = buildcircuit(ψ,meas_gates)
    ψ_out = runcircuit(ψ,meas_gates)
    x1 = PastaQ.getsamples!(ψ_out)
    x1 .+= 1 
    
    if (basis[1] == "Z")
      ψ1 = ψ_out[1] * setelt(s[1]=>x1[1])
    else
      rotation = gate(ψ_out,"basis$(basis[1])",1; dag = true)
      ψ_r = ψ_out[1] * rotation
      ψ1 = noprime!(ψ_r) * setelt(s[1]=>x1[1])
    end
    for j in 2:N-1
      if (basis[j] == "Z")
        ψ1 = ψ1 * ψ_out[j] * setelt(s[j]=>x1[j])
      else
        rotation = gate(ψ_out,"basis$(basis[j])",j; dag = true)
        ψ_r = ψ_out[j] * rotation
        ψ1 = ψ1 * noprime!(ψ_r) * setelt(s[j]=>x1[j])
      end
    end
    if (basis[N] == "Z")
      ψ1 = (ψ1 * ψ_out[N] * setelt(s[N]=>x1[N]))[]
    else
      rotation = gate(ψ_out,"basis$(basis[N])",N, dag = true)
      ψ_r = ψ_out[N] * rotation
      ψ1 = (ψ1 * noprime!(ψ_r) * setelt(s[N]=>x1[N]))[]
    end
    
    # Change format of data
    x2 = []
    for j in 1:N
      if basis[j] == "X"
        if x1[j] == 1
          push!(x2,"X+")
        else
          push!(x2,"X-")
        end
      elseif basis[j] == "Y"
        if x1[j] == 1
          push!(x2,"Y+")
        else
          push!(x2,"Y-")
        end
      elseif basis[j] == "Z"
        if x1[j] == 1
          push!(x2,"Z+")
        else
          push!(x2,"Z-")
        end
      end
    end
  
    ψ2 = ψ_out[1] * dag(state(x2[1],s[1]))
    for j in 2:N
      ψ_r = ψ_out[j] * dag(state(x2[j],s[j]))
      ψ2 = ψ2 * ψ_r
    end
    ψ2 = ψ2[]
    @test ψ1 ≈ ψ2

    if (basis[1] == "Z")
      ψ1 = dag(ψ_out[1]) * setelt(s[1]=>x1[1])
    else
      rotation = gate(ψ_out,"basis$(basis[1])",1; dag = true)
      ψ_r = dag(ψ_out[1]) * dag(rotation)
      ψ1 = noprime!(ψ_r) * setelt(s[1]=>x1[1])
    end
    for j in 2:N-1
      if (basis[j] == "Z")
        ψ1 = ψ1 * dag(ψ_out[j]) * setelt(s[j]=>x1[j])
      else
        rotation = gate(ψ_out,"basis$(basis[j])",j; dag = true)
        ψ_r = dag(ψ_out[j]) * dag(rotation)
        ψ1 = ψ1 * noprime!(ψ_r) * setelt(s[j]=>x1[j])
      end
    end
    if (basis[N] == "Z")
      ψ1 = (ψ1 * dag(ψ_out[N]) * setelt(s[N]=>x1[N]))[]
    else
      rotation = gate(ψ_out,"basis$(basis[N])",N; dag = true)
      ψ_r = dag(ψ_out[N]) * dag(rotation)
      ψ1 = (ψ1 * noprime!(ψ_r) * setelt(s[N]=>x1[N]))[]
    end
  
    ψ2 = dag(ψ_out[1]) * state(x2[1],s[1])
    for j in 2:N
      ψ_r = dag(ψ_out[j]) * state(x2[j],s[j])
      ψ2 = ψ2 * ψ_r
    end
    ψ2 = ψ2[]
    @test ψ1 ≈ ψ2

  end
end

@testset "project unitary" begin
  N = 4
  ntrial=100
  gates = randomcircuit(N,4; layered = false)
 
  U = runcircuit(N,gates;process=true)
  
  bases = randombases(N,ntrial)
  preps = PastaQ.randompreparations(N,ntrial)
  
  for n in 1:ntrial
    mgates = PastaQ.measurementgates(bases[n,:])
    ψ_in  = productstate(N, preps[n,:])
    ψ_out = runcircuit(ψ_in,gates)
    
    Ψ_out = PastaQ.projectunitary(U,preps[n,:])
    @test PastaQ.array(ψ_out) ≈ PastaQ.array(Ψ_out) 
    
    ψ_m   = runcircuit(ψ_out,mgates)
    Ψ_m   = runcircuit(Ψ_out,mgates)
    @test PastaQ.array(ψ_m) ≈ PastaQ.array(Ψ_m) 
  end
  
end


@testset "choi matrix + projectchoi" begin
  N = 4
  ntrial = 100
  gates = randomcircuit(N, 4; layered = false)
 
  
  Λ = runcircuit(N, gates; process = true, noise = ("amplitude_damping", (γ = 0.1,)))
  
  bases = randombases(N,ntrial)
  preps = PastaQ.randompreparations(N,ntrial)
  for n in 1:ntrial
    mgates = PastaQ.measurementgates(bases[n,:])
    ψ_in  = productstate(N, preps[n,:])
    ρ_out = runcircuit(ψ_in, gates; noise = ("amplitude_damping", (γ = 0.1,)))
    
    Λ_out = PastaQ.projectchoi(Λ,preps[n,:])
    @test PastaQ.array(ρ_out) ≈ PastaQ.array(Λ_out) atol = 1e-6
    
    ρ_m   = runcircuit(ρ_out,mgates)
    Λ_m   = runcircuit(Λ_out,mgates)
    @test PastaQ.array(ρ_m) ≈ PastaQ.array(Λ_m) atol = 1e-6 
  end
end

@testset "getsamples" begin
  
  N = 4
  nshots = 10
  gates = randomcircuit(N, 4)
  ψ = runcircuit(N, gates)
  ρ = runcircuit(N, gates; noise = ("amplitude_damping", (γ = 0.1,)))

  # 1a) Generate data with a MPS on the reference basis
  data = PastaQ.getsamples!(ψ,nshots)
  @test size(data) == (nshots,N)
  # 1b) Generate data with a MPO on the reference basis
  data = PastaQ.getsamples!(ρ,nshots)
  @test size(data) == (nshots,N)
  
  # 2a) Generate data with a MPS on multiple bases
  bases = randombases(N,nshots;local_basis=["X","Y","Z"])
  data = getsamples(ψ,bases)
  @test size(data) == (nshots,N)
  # 2b) Generate data with a MPO on multiple bases
  bases = randombases(N,nshots;local_basis=["X","Y","Z"])
  data = getsamples(ρ,bases)
  @test size(data) == (nshots,N)

  # 3) Measure MPS at the output of a circuit
  data, _ = getsamples(N, gates, nshots)
  @test size(data) == (nshots, N)
  data, _ = getsamples(N, gates, nshots;
                       noise = ("amplitude_damping", (γ = 0.1,)))
  @test size(data) == (nshots, N)
  data, _ = getsamples(N, gates, nshots; local_basis = ["X","Y","Z"])
  @test size(data) == (nshots,N)
  data, _ = getsamples(N, gates, nshots;
                       noise = ("amplitude_damping", (γ = 0.1,)),
                       local_basis = ["X","Y","Z"])
  @test size(data) == (nshots,N)
  data, M = getsamples(N, gates, nshots;)
  data, M = getsamples(N, gates, nshots; noise = ("amplitude_damping", (γ = 0.1,)))
  data, M = getsamples(N, gates, nshots; local_basis = ["X","Y","Z"])
  data, M = getsamples(N, gates, nshots;
                       noise = ("amplitude_damping", (γ = 0.1,)),
                       local_basis = ["X","Y","Z"])
  
  # 4) Process tomography
  data, _ = getsamples(N, gates, nshots; process = true, build_process = false)
  @test size(data) == (nshots,N)
  data, _ = getsamples(N, gates, nshots;
                       process = true,
                       build_process = false,
                       noise = ("amplitude_damping", (γ = 0.1,)))
  @test size(data) == (nshots,N)
  data, Λ  = getsamples(N, gates, nshots; process = true, build_process = true)
  @test Λ isa MPO
  data, Λ = getsamples(N, gates, nshots;
                       process = true,
                       build_process = true,
                       noise = ("amplitude_damping", (γ = 0.1,)))
  @test PastaQ.ischoi(Λ) == true #isa Choi{MPO}

end

@testset "readout errors" begin
  
  N = 4
  nshots = 10
  gates = randomcircuit(N,4)
  
  readout_errors = (p1given0 = 0.01, p0given1 = 0.04)
  ψ = runcircuit(N, gates)
  ρ = runcircuit(N, gates; noise = ("amplitude_damping", (γ = 0.1,)))
  
  # 1a) Generate data with a MPS on the reference basis
  data = PastaQ.getsamples!(ψ,nshots;readout_errors = readout_errors)
  @test size(data) == (nshots,N)
  # 1b) Generate data with a MPO on treadout_errors = readout_errors
  data = PastaQ.getsamples!(ρ,nshots;readout_errors = readout_errors)
  @test size(data) == (nshots,N)
  
  # 2a) Generate data with a MPS on multiple bases
  bases = randombases(N,nshots;local_basis=["X","Y","Z"])
  data = getsamples(ψ,bases;readout_errors = readout_errors)
  @test size(data) == (nshots,N)
  # 2b) Generate data with a MPO on multiple bases
  bases = randombases(N,nshots;local_basis=["X","Y","Z"])
  data = getsamples(ρ,bases;readout_errors = readout_errors)
  @test size(data) == (nshots,N)

  # 3) Measure MPS at the output of a circuit
  data, _ = getsamples(N, gates, nshots; readout_errors = readout_errors)
  @test size(data) == (nshots,N)
  data, _ = getsamples(gates, nshots; readout_errors = readout_errors)
  @test size(data) == (nshots,N)
  data, _ = getsamples(N, gates, nshots;
                       noise = ("amplitude_damping", (γ = 0.1,)),
                       readout_errors = readout_errors)
  @test size(data) == (nshots,N)
  data, _ = getsamples(N, gates, nshots; local_basis = ["X","Y","Z"], readout_errors = readout_errors)
  @test size(data) == (nshots,N)
  data, _ = getsamples(gates, nshots; local_basis = ["X","Y","Z"], readout_errors = readout_errors)
  @test size(data) == (nshots,N)
  data, _ = getsamples(N, gates, nshots;
                       noise = ("amplitude_damping", (γ = 0.1,)),
                       local_basis = ["X","Y","Z"],
                       readout_errors = readout_errors)
  @test size(data) == (nshots,N)
  data, _ = getsamples(gates, nshots;
                       noise = ("amplitude_damping", (γ = 0.1,)),
                       local_basis = ["X","Y","Z"],
                       readout_errors = readout_errors)
  @test size(data) == (nshots,N)
  data, M = getsamples(N, gates, nshots; readout_errors = readout_errors)
  data, M = getsamples(N, gates, nshots;
                       noise = ("amplitude_damping", (γ = 0.1,)), readout_errors = readout_errors)
  data, M = getsamples(N, gates, nshots; local_basis = ["X","Y","Z"], readout_errors = readout_errors)
  data, M = getsamples(N, gates, nshots;
                       noise = ("amplitude_damping", (γ=0.1,)),
                       local_basis = ["X","Y","Z"],
                       readout_errors = readout_errors)
  
  # 4) Process tomography
  data, _ = getsamples(N, gates, nshots;
                       process = true,
                       build_process = false,
                       readout_errors = readout_errors)
  @test size(data) == (nshots,N)
  data, _ = getsamples(gates, nshots;
                       process = true,
                       build_process = false,
                       readout_errors = readout_errors)
  @test size(data) == (nshots,N)
  data, _ = getsamples(N, gates, nshots;
                       process = true,
                       build_process = false,
                       noise = ("amplitude_damping", (γ = 0.1,)),
                       readout_errors = readout_errors)
  @test size(data) == (nshots,N)
  data, Λ = getsamples(N, gates, nshots;
                       process = true,
                       build_process = true,
                       readout_errors = readout_errors)
  data, Λ = getsamples(N, gates, nshots;
                       process = true,
                       build_process = true,
                       noise = ("amplitude_damping", (γ = 0.1,)),
                       readout_errors = readout_errors)
end
