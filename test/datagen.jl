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
  samples = generatedata!(ψ,nshots)
  @test size(samples)[1] == nshots
  @test size(samples)[2] == N
  data_prob = empiricalprobability(samples)
  @test probs ≈ data_prob atol=1e-2

  ρ = runcircuit(N,gates,noise="AD",γ=0.01)
  ρ_mat = fullmatrix(ρ)
  probs = real(diag(ρ_mat))

  samples = generatedata!(ρ,nshots)
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
    x1 = generatedata!(ψ_out)
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



@testset "project choi" begin
  N = 4
  ntrial=100
  gates = randomcircuit(N,4)
  #gates = Tuple[("Rn",1,(θ=1.0,ϕ=2.0,λ=3.0))]
 
  # 1: Noiseless
  Ψ = choimatrix(N,gates)
  
  bases = randombases(N,ntrial)
  preps = randompreparations(N,ntrial)
  
  for n in 1:ntrial
    pgates = preparationgates(preps[n,:])
    mgates = measurementgates(bases[n,:])
    ψ_in  = runcircuit(N,pgates)
    ψ_out = runcircuit(ψ_in,gates)
    
    Ψ_out = projectchoi(Ψ,preps[n,:])
    @test fullvector(ψ_out) ≈ fullvector(Ψ_out) 
    
    ψ_m   = runcircuit(ψ_out,mgates)
    Ψ_m   = runcircuit(Ψ_out,mgates)
    @test fullvector(ψ_m) ≈ fullvector(Ψ_m) 
  end
  
  # 2: Noisy
  Λ = choimatrix(N,gates;noise="AD",γ=0.1)
  
  bases = randombases(N,ntrial)
  preps = randompreparations(N,ntrial)
  for n in 1:ntrial
    pgates = preparationgates(preps[n,:])
    mgates = measurementgates(bases[n,:])
    ψ_in  = runcircuit(N,pgates)
    ρ_out = runcircuit(ψ_in,gates;noise="AD",γ=0.1)
    
    Λ_out = projectchoi(Λ,preps[n,:])
    @test fullmatrix(ρ_out) ≈ fullmatrix(Λ_out)
    
    ρ_m   = runcircuit(ρ_out,mgates)
    Λ_m   = runcircuit(Λ_out,mgates)
    @test fullmatrix(ρ_m) ≈ fullmatrix(Λ_m) 
  end
end

@testset " choi matrix" begin
  N = 1
  ψ0 = qubits(N)
  gates = Tuple[("Rn",1,(θ=1.0,ϕ=2.0,λ=3.0))]

  gate_tensors = compilecircuit(ψ0,gates)
  
  Φ = choimatrix(N,gates)
  
  prep  = ["X+","X-","Y+","Y-","Z+","Z-"]
  bases = ["X","Y","Z"]
  
  conditional_probability = zeros(6,6)

  counter_a = 1
  for j in prep
    a = [j]
    p_gates = preparationgates(a)
    ψ_in   = runcircuit(ψ0,p_gates)
    ψ_out  = runcircuit(ψ_in,gate_tensors) 
    
    counter_b = 1
    for k in bases
      b = [k]
      m_gates = measurementgates(b)
      ψ_meas = runcircuit(ψ_out,m_gates) 
      
      prob1 = abs2.(fullvector(ψ_meas))
      
      phidag = dag(Φ)
      s = siteinds(Φ)
      
      for (i,σ) in enumerate(["+","-"])
        input_state  = "state" * a[1]
        output_state = "state" * b[1] * σ 
        tmp = phidag[1] * dag(gate(input_state,s[1]))
        tmp = tmp * phidag[2]
        psix = tmp * gate(output_state,s[2])
        prob2 = abs2(psix[])
        #print("P=",a[1],"  ","M=",b[1]*σ,"  ")
        #println("p1=",prob1[i],"  ","p2=",prob2)
        @test prob1[i] ≈ prob2 atol=1e-1
        
        conditional_probability[counter_a,counter_b] = prob2
        counter_b += 1
      end
      
    end
    counter_a += 1
  end
  
  #nshots = 10000
  #(data_in,data_out) = generatedata(N,gates,nshots;process=true)
  #
  #emp_prob = probability_of(data_in,data_out,["X+"],["X+"])
  #@show emp_prob
  #counter_a = 1
  #for j in prep
  #  a = [j]
  #  counter_b = 1
  #  for k in prep
  #    b = [k]
  #    emp_prob = 18*probability_of(data_in,data_out,a,b)
  #    @show a,b,emp_prob,conditional_probability[counter_a,counter_b]
  #    counter_b += 1
  #    #@show a,b
  #  end
  #  counter_a += 1
  #end
end

@testset "generatedata" begin
  
  N = 4
  nshots = 100
  gates = randomcircuit(N,4)
  ψ = runcircuit(N,gates)
  ρ = runcircuit(N,gates;noise="AD",γ=0.1)

  # 1a) Generate data with a MPS on the reference basis
  data = generatedata!(ψ,nshots)
  @test size(data) == (nshots,N)
  # 1b) Generate data with a MPO on the reference basis
  data = generatedata!(ρ,nshots)
  @test size(data) == (nshots,N)
  
  # 2a) Generate data with a MPS on multiple bases
  bases = randombases(N,nshots;localbasis=["X","Y","Z"])
  data = generatedata(ψ,bases)
  @test size(data) == (nshots,N)
  # 2b) Generate data with a MPO on multiple bases
  bases = randombases(N,nshots;localbasis=["X","Y","Z"])
  data = generatedata(ρ,bases)
  @test size(data) == (nshots,N)

  # 3) Measure MPS at the output of a circuit
  data = generatedata(N,gates,nshots)
  @test size(data) == (nshots,N)
  data = generatedata(N,gates,nshots;noise="AD",γ=0.1)
  @test size(data) == (nshots,N)
  data = generatedata(N,gates,nshots;localbasis=["X","Y","Z"])
  @test size(data) == (nshots,N)
  data = generatedata(N,gates,nshots;noise="AD",γ=0.1,localbasis=["X","Y","Z"])
  @test size(data) == (nshots,N)
  M,data = generatedata(N,gates,nshots;return_state=true)
  M,data = generatedata(N,gates,nshots;return_state=true,noise="AD",γ=0.1)
  M,data = generatedata(N,gates,nshots;return_state=true,localbasis=["X","Y","Z"])
  M,data = generatedata(N,gates,nshots;return_state=true,noise="AD",γ=0.1,localbasis=["X","Y","Z"])
  
  # 4) Process tomography
  (data_in,data_out) = generatedata(N,gates,nshots;process=true,choi=false)
  @test size(data_in) == (nshots,N)
  @test size(data_out) == (nshots,N)
  (data_in,data_out) = generatedata(N,gates,nshots;process=true,choi=false,noise="AD",γ=0.1)
  @test size(data_in) == (nshots,N)
  @test size(data_out) == (nshots,N)
  (Λ,data_in,data_out) = generatedata(N,gates,nshots;process=true,choi=true,return_state=true)
  (Λ,data_in,data_out) = generatedata(N,gates,nshots;process=true,choi=true,return_state=true,noise="AD",γ=0.1)

end
