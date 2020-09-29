using PastaQ
using ITensors
using HDF5
using JLD
using Test
using LinearAlgebra

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

function probability_of(data_in::Array,data_out::Array,target_in::Array,target_out::Array)
  nshots = size(data_in)[1]
  prob = 0.0
  for n in 1:nshots
    if data_in[n,:]==target_in && data_out[n,:]==target_out
      prob += 1.0/nshots
    end
  end
  return prob
end

@testset "generation of preparation states" begin
  N = 4
  nshots = 100
  states = randompreparations(N,nshots)
  @test size(states)[1] == nshots
  @test size(states)[2] == N
  
  states = randompreparations(N,nshots,n_distinctstates=10)
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
  bases = randombases(N,nshots)
  @test size(bases)[1] == nshots
  @test size(bases)[2] == N
  
  bases = randombases(N,nshots,n_distinctbases=10)
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
  ψ0 = qubits(N)
  gates = randomcircuit(N,depth)
  ψ = runcircuit(ψ0,gates)
  ψ_vec = fullvector(ψ)
  probs = abs2.(ψ_vec)
  
  nshots = 100000
  samples = getsamples!(ψ,nshots)
  @test size(samples)[1] == nshots
  @test size(samples)[2] == N
  data_prob = empiricalprobability(samples)
  @test probs ≈ data_prob atol=1e-2

  ρ = runcircuit(N,gates,noise="AD",γ=0.01)
  ρ_mat = fullmatrix(ρ)
  probs = real(diag(ρ_mat))

  samples = getsamples!(ρ,nshots)
  @test size(samples)[1] == nshots
  @test size(samples)[2] == N
  data_prob = empiricalprobability(samples)
  @test probs ≈ data_prob atol=1e-2
end


@testset "measurement projections" begin
  N = 8
  nshots = 20
  ψ0 = qubits(N)
  bases = randombases(N,nshots)
  
  depth = 8
  gates = randomcircuit(N,depth)
  ψ = runcircuit(ψ0,gates)
  s = siteinds(ψ)

  for n in 1:nshots
    basis = bases[n,:]
    meas_gates = measurementgates(basis)
    #meas_tensors = compilecircuit(ψ,meas_gates)
    ψ_out = runcircuit(ψ,meas_gates)
    x1 = getsamples!(ψ_out)
    x1 .+= 1 
    
    if (basis[1] == "Z")
      ψ1 = ψ_out[1] * setelt(s[1]=>x1[1])
    else
      rotation = gate(ψ_out,"meas$(basis[1])",1)
      ψ_r = ψ_out[1] * rotation
      ψ1 = noprime!(ψ_r) * setelt(s[1]=>x1[1])
    end
    for j in 2:N-1
      if (basis[j] == "Z")
        ψ1 = ψ1 * ψ_out[j] * setelt(s[j]=>x1[j])
      else
        rotation = gate(ψ_out,"meas$(basis[j])",j)
        ψ_r = ψ_out[j] * rotation
        ψ1 = ψ1 * noprime!(ψ_r) * setelt(s[j]=>x1[j])
      end
    end
    if (basis[N] == "Z")
      ψ1 = (ψ1 * ψ_out[N] * setelt(s[N]=>x1[N]))[]
    else
      rotation = gate(ψ_out,"meas$(basis[N])",N)
      ψ_r = ψ_out[N] * rotation
      ψ1 = (ψ1 * noprime!(ψ_r) * setelt(s[N]=>x1[N]))[]
    end
    
    # Change format of data
    x2 = []
    for j in 1:N
      if basis[j] == "X"
        if x1[j] == 1
          push!(x2,"stateX+")
        else
          push!(x2,"stateX-")
        end
      elseif basis[j] == "Y"
        if x1[j] == 1
          push!(x2,"stateY+")
        else
          push!(x2,"stateY-")
        end
      elseif basis[j] == "Z"
        if x1[j] == 1
          push!(x2,"stateZ+")
        else
          push!(x2,"stateZ-")
        end
      end
    end
  
    ψ2 = ψ_out[1] * dag(gate(x2[1],s[1]))
    for j in 2:N
      ψ_r = ψ_out[j] * dag(gate(x2[j],s[j]))
      ψ2 = ψ2 * ψ_r
    end
    ψ2 = ψ2[]
    @test ψ1 ≈ ψ2

    if (basis[1] == "Z")
      ψ1 = dag(ψ_out[1]) * setelt(s[1]=>x1[1])
    else
      rotation = gate(ψ_out,"meas$(basis[1])",1)
      ψ_r = dag(ψ_out[1]) * dag(rotation)
      ψ1 = noprime!(ψ_r) * setelt(s[1]=>x1[1])
    end
    for j in 2:N-1
      if (basis[j] == "Z")
        ψ1 = ψ1 * dag(ψ_out[j]) * setelt(s[j]=>x1[j])
      else
        rotation = gate(ψ_out,"meas$(basis[j])",j)
        ψ_r = dag(ψ_out[j]) * dag(rotation)
        ψ1 = ψ1 * noprime!(ψ_r) * setelt(s[j]=>x1[j])
      end
    end
    if (basis[N] == "Z")
      ψ1 = (ψ1 * dag(ψ_out[N]) * setelt(s[N]=>x1[N]))[]
    else
      rotation = gate(ψ_out,"meas$(basis[N])",N)
      ψ_r = dag(ψ_out[N]) * dag(rotation)
      ψ1 = (ψ1 * noprime!(ψ_r) * setelt(s[N]=>x1[N]))[]
    end
  
    ψ2 = dag(ψ_out[1]) * gate(x2[1],s[1])
    for j in 2:N
      ψ_r = dag(ψ_out[j]) * gate(x2[j],s[j])
      ψ2 = ψ2 * ψ_r
    end
    ψ2 = ψ2[]
    @test ψ1 ≈ ψ2

  end
end

@testset "project unitary" begin
  N = 4
  ntrial=100
  gates = randomcircuit(N,4)
 
  U = runcircuit(N,gates;process=true)
  
  bases = randombases(N,ntrial)
  preps = randompreparations(N,ntrial)
  
  for n in 1:ntrial
    pgates = preparationgates(preps[n,:])
    mgates = measurementgates(bases[n,:])
    ψ_in  = runcircuit(N,pgates)
    ψ_out = runcircuit(ψ_in,gates)
    
    Ψ_out = PastaQ.projectunitary(U,preps[n,:])
    @test fullvector(ψ_out) ≈ fullvector(Ψ_out) 
    
    ψ_m   = runcircuit(ψ_out,mgates)
    Ψ_m   = runcircuit(Ψ_out,mgates)
    @test fullvector(ψ_m) ≈ fullvector(Ψ_m) 
  end
  
end


@testset "choi matrix + projectchoi" begin
  N = 4
  ntrial=100
  gates = randomcircuit(N,4)
  
  Λ = runcircuit(N,gates;process=true,noise="AD",γ=0.1)
  
  bases = randombases(N,ntrial)
  preps = randompreparations(N,ntrial)
  for n in 1:ntrial
    pgates = preparationgates(preps[n,:])
    mgates = measurementgates(bases[n,:])
    ψ_in  = runcircuit(N,pgates)
    ρ_out = runcircuit(ψ_in,gates;noise="AD",γ=0.1)
    
    Λ_out = PastaQ.projectchoi(Λ,preps[n,:])
    @test fullmatrix(ρ_out) ≈ fullmatrix(Λ_out)
    
    ρ_m   = runcircuit(ρ_out,mgates)
    Λ_m   = runcircuit(Λ_out,mgates)
    @test fullmatrix(ρ_m) ≈ fullmatrix(Λ_m) 
  end
end

@testset "getsamples" begin
  
  N = 4
  nshots = 10
  gates = randomcircuit(N,4)
  ψ = runcircuit(N,gates)
  ρ = runcircuit(N,gates;noise="AD",γ=0.1)

  # 1a) Generate data with a MPS on the reference basis
  data = getsamples!(ψ,nshots)
  @test size(data) == (nshots,N)
  # 1b) Generate data with a MPO on the reference basis
  data = getsamples!(ρ,nshots)
  @test size(data) == (nshots,N)
  
  # 2a) Generate data with a MPS on multiple bases
  bases = randombases(N,nshots;localbasis=["X","Y","Z"])
  data = getsamples(ψ,bases)
  @test size(data) == (nshots,N)
  # 2b) Generate data with a MPO on multiple bases
  bases = randombases(N,nshots;localbasis=["X","Y","Z"])
  data = getsamples(ρ,bases)
  @test size(data) == (nshots,N)

  # 3) Measure MPS at the output of a circuit
  data = getsamples(N,gates,nshots)
  @test size(data) == (nshots,N)
  data = getsamples(N,gates,nshots;noise="AD",γ=0.1)
  @test size(data) == (nshots,N)
  data = getsamples(N,gates,nshots;localbasis=["X","Y","Z"])
  @test size(data) == (nshots,N)
  data = getsamples(N,gates,nshots;noise="AD",γ=0.1,localbasis=["X","Y","Z"])
  @test size(data) == (nshots,N)
  M,data = getsamples(N,gates,nshots;return_state=true)
  M,data = getsamples(N,gates,nshots;return_state=true,noise="AD",γ=0.1)
  M,data = getsamples(N,gates,nshots;return_state=true,localbasis=["X","Y","Z"])
  M,data = getsamples(N,gates,nshots;return_state=true,noise="AD",γ=0.1,localbasis=["X","Y","Z"])
  
  # 4) Process tomography
  (data_in,data_out) = getsamples(N,gates,nshots;process=true,build_process=false)
  @test size(data_in) == (nshots,N)
  @test size(data_out) == (nshots,N)
  (data_in,data_out) = getsamples(N,gates,nshots;process=true,build_process=false,noise="AD",γ=0.1)
  @test size(data_in) == (nshots,N)
  @test size(data_out) == (nshots,N)
  (Λ,data_in,data_out) = getsamples(N,gates,nshots;process=true,build_process=true,return_state=true)
  (Λ,data_in,data_out) = getsamples(N,gates,nshots;process=true,build_process=true,return_state=true,noise="AD",γ=0.1)

end

@testset "readout errors" begin
  
  N = 4
  nshots = 10
  gates = randomcircuit(N,4)
  ψ = runcircuit(N,gates)
  ρ = runcircuit(N,gates;noise="AD",γ=0.1)
  
  # 1a) Generate data with a MPS on the reference basis
  data = getsamples!(ψ,nshots;readout_errors = [0.01,0.04])
  @test size(data) == (nshots,N)
  # 1b) Generate data with a MPO on the reference basis
  data = getsamples!(ρ,nshots;readout_errors = [0.01,0.04])
  @test size(data) == (nshots,N)
  
  # 2a) Generate data with a MPS on multiple bases
  bases = randombases(N,nshots;localbasis=["X","Y","Z"])
  data = getsamples(ψ,bases;readout_errors = [0.01,0.04])
  @test size(data) == (nshots,N)
  # 2b) Generate data with a MPO on multiple bases
  bases = randombases(N,nshots;localbasis=["X","Y","Z"])
  data = getsamples(ρ,bases;readout_errors = [0.01,0.04])
  @test size(data) == (nshots,N)

  # 3) Measure MPS at the output of a circuit
  data = getsamples(N,gates,nshots;readout_errors = [0.01,0.04])
  @test size(data) == (nshots,N)
  data = getsamples(N,gates,nshots;noise="AD",γ=0.1,readout_errors = [0.01,0.04])
  @test size(data) == (nshots,N)
  data = getsamples(N,gates,nshots;localbasis=["X","Y","Z"],readout_errors = [0.01,0.04])
  @test size(data) == (nshots,N)
  data = getsamples(N,gates,nshots;noise="AD",γ=0.1,localbasis=["X","Y","Z"],readout_errors = [0.01,0.04])
  @test size(data) == (nshots,N)
  M,data = getsamples(N,gates,nshots;return_state=true,readout_errors = [0.01,0.04])
  M,data = getsamples(N,gates,nshots;return_state=true,noise="AD",γ=0.1,readout_errors = [0.01,0.04])
  M,data = getsamples(N,gates,nshots;return_state=true,localbasis=["X","Y","Z"],readout_errors = [0.01,0.04])
  M,data = getsamples(N,gates,nshots;return_state=true,noise="AD",γ=0.1,localbasis=["X","Y","Z"],readout_errors = [0.01,0.04])
  
  # 4) Process tomography
  (data_in,data_out) = getsamples(N,gates,nshots;process=true,build_process=false,readout_errors = [0.01,0.04])
  @test size(data_in) == (nshots,N)
  @test size(data_out) == (nshots,N)
  (data_in,data_out) = getsamples(N,gates,nshots;process=true,build_process=false,noise="AD",γ=0.1,readout_errors = [0.01,0.04])
  @test size(data_in) == (nshots,N)
  @test size(data_out) == (nshots,N)
  (Λ,data_in,data_out) = getsamples(N,gates,nshots;process=true,build_process=true,return_state=true,readout_errors = [0.01,0.04])
  (Λ,data_in,data_out) = getsamples(N,gates,nshots;process=true,build_process=true,return_state=true,noise="AD",γ=0.1,readout_errors = [0.01,0.04])

end
