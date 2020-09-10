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


@testset "generation of preparation states" begin
  N = 4
  nshots = 100
  states = preparationsettings(N,nshots)
  @test size(states)[1] == nshots
  @test size(states)[2] == N
  
  states = preparationsettings(N,nshots,numprep=10)
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
  bases = measurementsettings(N,nshots)
  @test size(bases)[1] == nshots
  @test size(bases)[2] == N
  
  bases = measurementsettings(N,nshots,numbases=10)
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
  gates = randomquantumcircuit(N,depth)
  ψ = runcircuit(ψ0,gates)
  ψ_vec = fullvector(ψ)
  probs = abs2.(ψ_vec)
  
  nshots = 100000
  samples = measure(ψ,nshots)
  @test size(samples)[1] == nshots
  @test size(samples)[2] == N
  data_prob = empiricalprobability(samples)
  @test probs ≈ data_prob atol=1e-2

  ρ = runcircuit(N,gates,noise="AD",γ=0.01)
  ρ_mat = fullmatrix(ρ)
  probs = real(diag(ρ_mat))

  samples = measure(ρ,nshots)
  @test size(samples)[1] == nshots
  @test size(samples)[2] == N
  data_prob = empiricalprobability(samples)
  @test probs ≈ data_prob atol=1e-2
end

@testset "measurement projections" begin
  N = 8
  nshots = 20
  ψ0 = qubits(N)
  bases = measurementsettings(N,nshots)
  
  depth = 8
  gates = randomquantumcircuit(N,depth)
  ψ = runcircuit(ψ0,gates)
  s = siteinds(ψ)

  for n in 1:nshots
    basis = bases[n,:]
    meas_gates = measurementgates(basis)
    #meas_tensors = compilecircuit(ψ,meas_gates)
    ψ_out = runcircuit(ψ,meas_gates)
    x1 = measure(ψ_out,1)
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

