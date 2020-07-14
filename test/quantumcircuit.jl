using PastaQ
using ITensors
using HDF5
using JLD
using Test
using LinearAlgebra

function runcircuitFULL(N::Int,tensors::Array)
  """ Assumes NN gates, and for 2q gates-> [q+1,q] """
  ngates = length(tensors)
  id_mat = [1. 0.;0. 1.]
  swap   = [1 0 0 0;
            0 0 1 0;
            0 1 0 0;
            0 0 0 1]
  U = 1.0
  for j in 1:N
    U = kron(U,id_mat)
  end
  for tensor in tensors
    # 1q gate
    if (length(inds(tensor)) == 2)
      site = getsitenumber(firstind(tensor,"Site"))
      u = 1.0
      for j in 1:N
        if (j == site)
          u = kron(u,array(tensor))
        else
          u = kron(u,id_mat)
        end
      end
      U = u * U
    #2q gate
    else
      site1 = getsitenumber(inds(tensor,plev=1)[2])
      site2 = getsitenumber(inds(tensor,plev=1)[1])
      site = min(site1,site2)
      if (site1<site2)
        gate = reshape(array(tensor),(4,4))
      else
        gate = swap * reshape(array(tensor),(4,4)) * swap
      end
      # NN 2q gate
      if abs(site1-site2) == 1
        u = 1.0
        #if (site1<site2)
        #  gate = reshape(array(tensor),(4,4))
        #else
        #  gate = swap * reshape(array(tensor),(4,4)) * swap
        #  #gate = swap * gate
        #end
        for j in 1:N-1
          if (j == site)
            u = kron(u,gate)
          else
            u = kron(u,id_mat)
          end
        end
        U = u * U
      else
        nswaps = abs(site1-site2)-1
        if site1 > site2
          start = site2
        else
          start = site1
        end
        # Swap
        for n in 1:nswaps
          u = 1.0
          for j in 1:N-1
            if j == start+n-1
              u = kron(u,swap)
            else
              u = kron(u,id_mat)
            end
          end
          U = u * U
        end
        # Gate
        u = 1.0
        for j in 1:N-1
          if j == start+nswaps
            u = kron(u,gate)
            #u = kron(u,reshape(array(tensor),(4,4)))
          else
            u = kron(u,id_mat)
          end
        end
        U = u * U
        # Unswap
        for n in 1:nswaps
          u = 1.0
          for j in 1:N-1
            if j == start+nswaps-n
              u = kron(u,swap)
            else
              u = kron(u,id_mat)
            end
          end
          U = u * U
        end
      end
    end
  end
  psi = U[:,1]
  return psi,U
end

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

@testset "qubits initialization" begin
  N = 1
  psi = qubits(N)
  @test length(psi) == 1
  @test length(inds(psi[1],"Link")) == 0
  @test fullvector(psi) ≈ [1, 0]
  N = 5
  psi = qubits(N)
  @test length(psi) == 5
  psi_vec = fullvector(psi)
  exact_vec = zeros(1<<N)
  exact_vec[1] = 1.0
  @test psi_vec ≈ exact_vec
end

@testset "circuit MPO initialization" begin
  N = 5
  U = circuit(N)
  @test length(U) == N
  U_mat = fullmatrix(U)
  exact_mat = Matrix{ComplexF64}(I, 1<<N, 1<<N)
  @test U_mat ≈ exact_mat
end

@testset "runcircuit: hadamardlayer N=10" begin
  N = 10 
  gates = []
  hadamardlayer!(gates,N)
  @test length(gates) == N
  psi = qubits(N)
  gate_tensors = compilecircuit(psi,gates)
  @test length(gate_tensors) == N
  runcircuit!(psi,gate_tensors)
  
  U = circuit(N)
  gate_tensors = compilecircuit(U,gates)
  runcircuit!(U,gate_tensors)
  
  exact_psi,exact_U = runcircuitFULL(N,gate_tensors)
  @test exact_psi ≈ fullvector(psi)
  @test exact_U ≈ fullmatrix(U)
end

@testset "runcircuit: rand1Qrotationlayer N=10" begin
  N = 10
  gates = []
  rand1Qrotationlayer!(gates,N)
  @test length(gates) == N

  psi0 = qubits(N)
  gate_tensors = compilecircuit(psi0, gates)
  @test length(gate_tensors) == N
  psi = runcircuit(psi0, gate_tensors)

  disable_warn_itensor_order!()
  @test prod(psi) ≈ noprime(prod(psi0) * prod(gate_tensors))
  reset_warn_itensor_order!()

  U0 = circuit(N)
  gate_tensors = compilecircuit(U0, gates)
  U = runcircuit(U0, gate_tensors)

  disable_warn_itensor_order!()
  @test prod(U) ≈ prod(gate_tensors)
  reset_warn_itensor_order!()
end

@testset "runcircuit: Cx layer N=10" begin
  N = 10
  gates = []
  Cxlayer!(gates,N,sequence = "odd") 
  @test length(gates) == N÷2
  psi = qubits(N)
  gate_tensors = compilecircuit(psi,gates)
  @test length(gate_tensors) == N÷2
  runcircuit!(psi,gate_tensors)

  U = circuit(N)
  gate_tensors = compilecircuit(U,gates)
  runcircuit!(U,gate_tensors)
  
  exact_psi,exact_U = runcircuitFULL(N,gate_tensors)
  @test exact_psi ≈ fullvector(psi)
  @test exact_U ≈ fullmatrix(U)

  gates = []
  Cxlayer!(gates,N,sequence = "even") 
  @test length(gates) == N÷2-1
  psi0 = qubits(N)
  gate_tensors = compilecircuit(psi0, gates)
  @test length(gate_tensors) == N÷2-1
  psi = runcircuit(psi0, gate_tensors)

  disable_warn_itensor_order!()
  @test prod(psi) ≈ noprime(prod(psi0) * prod(gate_tensors))
  reset_warn_itensor_order!()

  U0 = circuit(N)
  gate_tensors = compilecircuit(U0, gates)
  U = runcircuit(U0, gate_tensors)

  disable_warn_itensor_order!()
  # TODO: replace with mapprime([...], 2 => 1, 1 => 0)
  @test prod(U) ≈ mapprime(mapprime(prime(prod(U0)) * prod(gate_tensors), 1, 0), 2, 1)
  reset_warn_itensor_order!()
end

@testset "runcircuit: random quantum circuit" begin
  N = 10
  depth = 8
  gates = randomquantumcircuit(N,depth)
  ngates = N*depth + depth÷2 * (N-1)
  @test length(gates) == ngates
  psi0 = qubits(N)
  gate_tensors = compilecircuit(psi0, gates)
  @test length(gate_tensors) == ngates
  psi = runcircuit(psi0, gate_tensors)
  
  @test prod(psi) ≈ runcircuit(prod(psi0), gate_tensors)

  U0 = circuit(N)
  gate_tensors = compilecircuit(U0, gates)
  U = runcircuit(U0, gate_tensors)

  disable_warn_itensor_order!()
  @test prod(U) ≈ runcircuit(prod(U0), gate_tensors)
  reset_warn_itensor_order!()
end

@testset "runcircuit: inverted gate order" begin
  N = 10
  gates = randomquantumcircuit(N,2)
  
  for n in 1:10
    s1 = rand(2:N)
    s2 = s1-1
    push!(gates,(gate = "Cx", site = [s1,s2]))
  end
  psi0 = qubits(N)
  gate_tensors = compilecircuit(psi0, gates) 
  psi = runcircuit(psi0, gate_tensors)
  
  @test prod(psi) ≈ runcircuit(prod(psi0), gate_tensors)

  U0 = circuit(N)
  gate_tensors = compilecircuit(U0, gates)
  U = runcircuit(U0, gate_tensors)

  disable_warn_itensor_order!()
  @test prod(U) ≈ runcircuit(prod(U0), gate_tensors)
  reset_warn_itensor_order!()
end

@testset "runcircuit: long range gates" begin
  N = 10
  gates = randomquantumcircuit(N,2)
  
  for n in 1:10
    s1 = rand(1:N)
    s2 = rand(1:N)
    while s2 == s1
      s2 = rand(1:N)
    end
    @assert s1 != s2
    push!(gates,(gate = "Cx", site = [s1,s2]))
  end
  psi = qubits(N)
  gate_tensors = compilecircuit(psi,gates) 
  runcircuit!(psi,gate_tensors)
  
  exact_psi,exact_U = runcircuitFULL(N,gate_tensors)
  @test exact_psi ≈ fullvector(psi)
end

@testset "reset qubits" begin
  N = 5
  depth = 5
  gates = randomquantumcircuit(N,depth)
  psi = qubits(N)
  gate_tensors = compilecircuit(psi,gates)
  runcircuit!(psi,gate_tensors)
  
  psi = resetqubits!(psi)
  psi_vec = fullvector(psi)

  exact_vec = zeros(1<<N)
  exact_vec[1] = 1.0
  @test psi_vec ≈ exact_vec
end

@testset "generation of preparation states" begin
  N = 4
  nshots = 100
  states = generatepreparationsettings(N,nshots)
  @test size(states)[1] == nshots
  @test size(states)[2] == N
  
  states = generatepreparationsettings(N,nshots,numprep=10)
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
  bases = generatemeasurementsettings(N,nshots)
  @test size(bases)[1] == nshots
  @test size(bases)[2] == N
  
  bases = generatemeasurementsettings(N,nshots,numbases=10)
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
  psi = qubits(N)
  gates = randomquantumcircuit(N,depth)
  gate_tensors = compilecircuit(psi,gates)
  runcircuit!(psi,gate_tensors)
  psi_vec = fullvector(psi)
  prob = abs2.(psi_vec)
  
  nshots = 100000
  samples = measure(psi,nshots)
  @test size(samples)[1] == nshots
  @test size(samples)[2] == N
  data_prob = empiricalprobability(samples)
  @test prob ≈ data_prob atol=1e-2
end


@testset "measurement projections" begin
  N = 10
  nshots = 20
  psi = qubits(N)
  bases = generatemeasurementsettings(N,nshots)
  
  depth = 8
  gates = randomquantumcircuit(N,depth)
  gates = randomquantumcircuit(N,depth)
  gate_tensors = compilecircuit(psi,gates)
  runcircuit!(psi,gate_tensors)
  s = siteinds(psi)

  for n in 1:nshots
    basis = bases[n,:]
    meas_gates = makemeasurementgates(basis)
    meas_tensors = compilecircuit(psi,meas_gates)
    psi_out = runcircuit(psi,meas_tensors)
    x1 = measure(psi_out,1)
    x1 .+= 1 
    
    if (basis[1] == "Z")
      psi1 = psi_out[1] * setelt(s[1]=>x1[1])
    else
      rotation = makegate(psi_out,"m$(basis[1])",1)
      psi_r = psi_out[1] * rotation
      psi1 = noprime!(psi_r) * setelt(s[1]=>x1[1])
    end
    for j in 2:N-1
      if (basis[j] == "Z")
        psi1 = psi1 * psi_out[j] * setelt(s[j]=>x1[j])
      else
        rotation = makegate(psi_out,"m$(basis[j])",j)
        psi_r = psi_out[j] * rotation
        psi1 = psi1 * noprime!(psi_r) * setelt(s[j]=>x1[j])
      end
    end
    if (basis[N] == "Z")
      psi1 = (psi1 * psi_out[N] * setelt(s[N]=>x1[N]))[]
    else
      rotation = makegate(psi_out,"m$(basis[N])",N)
      psi_r = psi_out[N] * rotation
      psi1 = (psi1 * noprime!(psi_r) * setelt(s[N]=>x1[N]))[]
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
  
    psi2 = psi_out[1] * dag(measproj(x2[1],s[1]))
    for j in 2:N
      psi_r = psi_out[j] * dag(measproj(x2[j],s[j]))
      psi2 = psi2 * psi_r
    end
    psi2 = psi2[]
    @test psi1 ≈ psi2

    
    if (basis[1] == "Z")
      psi1 = dag(psi_out[1]) * setelt(s[1]=>x1[1])
    else
      rotation = makegate(psi_out,"m$(basis[1])",1)
      psi_r = dag(psi_out[1]) * dag(rotation)
      psi1 = noprime!(psi_r) * setelt(s[1]=>x1[1])
    end
    for j in 2:N-1
      if (basis[j] == "Z")
        psi1 = psi1 * dag(psi_out[j]) * setelt(s[j]=>x1[j])
      else
        rotation = makegate(psi_out,"m$(basis[j])",j)
        psi_r = dag(psi_out[j]) * dag(rotation)
        psi1 = psi1 * noprime!(psi_r) * setelt(s[j]=>x1[j])
      end
    end
    if (basis[N] == "Z")
      psi1 = (psi1 * dag(psi_out[N]) * setelt(s[N]=>x1[N]))[]
    else
      rotation = makegate(psi_out,"m$(basis[N])",N)
      psi_r = dag(psi_out[N]) * dag(rotation)
      psi1 = (psi1 * noprime!(psi_r) * setelt(s[N]=>x1[N]))[]
    end
  
    psi2 = dag(psi_out[1]) * measproj(x2[1],s[1])
    for j in 2:N
      psi_r = dag(psi_out[j]) * measproj(x2[j],s[j])
      psi2 = psi2 * psi_r
    end
    psi2 = psi2[]
    @test psi1 ≈ psi2

  end
end
