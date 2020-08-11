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
  ψ = qubits(N)
  @test length(ψ) == 1
  @test typeof(ψ) == MPS
  @test length(inds(ψ[1],"Link")) == 0
  @test fullvector(ψ) ≈ [1, 0]
  N = 5
  ψ = qubits(N)
  @test length(ψ) == 5
  ψ_vec = fullvector(ψ)
  exact_vec = zeros(1<<N)
  exact_vec[1] = 1.0
  @test ψ_vec ≈ exact_vec
end

@testset "circuit MPO initialization" begin
  N = 5
  U = circuit(N)
  @test length(U) == N
  U_mat = fullmatrix(U)
  exact_mat = Matrix{ComplexF64}(I, 1<<N, 1<<N)
  @test U_mat ≈ exact_mat
end

@testset "Density matrix initialization" begin
  N = 5
  ρ1 = qubits(N,mixed=true)
  @test length(ρ1) == N
  @test typeof(ρ1) == MPO
  ψ = qubits(N)
  ρ2 = qubits(N,mixed=true)
  @test fullmatrix(ρ1) ≈ fullmatrix(ρ2)
  exact_mat = zeros(1<<N,1<<N)
  exact_mat[1,1] = 1.0
  @test fullmatrix(ρ2) ≈ exact_mat
end

@testset "Choi matrix initialization" begin
  N = 5
  Λ = choi(N)
  @test length(Λ) == 2*N
  @test typeof(Λ) == MPS
  
  Λ = choi(N,mixed=true)
  @test length(Λ) == 2*N
  @test typeof(Λ) == MPO
end

@testset "reset qubits" begin
  N = 5
  depth = 5
  gates = randomquantumcircuit(N,depth)
  ψ0 = qubits(N)
  ψ = runcircuit(ψ0,gates)
  
  resetqubits!(ψ)
  psi_vec = fullvector(ψ)

  exact_vec = zeros(1<<N)
  exact_vec[1] = 1.0
  @test psi_vec ≈ exact_vec

  
  ρ0 = qubits(N,mixed=true)
  ρ = runcircuit(ρ0,gates)
  
  resetqubits!(ρ)
  ρ_mat = fullmatrix(ρ)

  exact_mat = zeros(1<<N,1<<N)
  exact_mat[1,1] = 1.0
  @test exact_mat ≈ ρ_mat
end


@testset "runcircuit: unitary quantum circuit" begin
  N = 3
  depth = 4
  gates = randomquantumcircuit(N,depth)
  ngates = N*depth + depth÷2 * (N-1)
  @test length(gates) == ngates
  
  #Pure state, noiseless circuit
  ψ0 = qubits(N)
  ψ = runcircuit(ψ0,gates)
  @test prod(ψ) ≈ runcircuit(prod(ψ0),compilecircuit(ψ0,gates))

  # Mixed state, noiseless circuit
  ρ0 = qubits(N,mixed=true) 
  ρ = runcircuit(ρ0,gates)
  @test prod(ρ) ≈ runcircuit(prod(ρ0),compilecircuit(ρ0,gates); apply_dag=true)
  
  U0 = circuit(N)
  U = runcircuit(U0,gates;get_unitary=true)
  disable_warn_order!()
  @test prod(U) ≈ runcircuit(prod(U0),compilecircuit(U0,gates))
end

@testset "runcircuit: noisy quantum circuit" begin
  N = 5
  depth = 4
  gates = randomquantumcircuit(N,depth)
  ngates = N*depth + depth÷2 * (N-1)
  @test length(gates) == ngates

  # Pure state, noisy circuit
  ψ0 = qubits(N)
  ρ = runcircuit(ψ0,gates,noise="DEP",p=0.1)
  ρ0 = densitymatrix(ψ0)
  disable_warn_order!()
  @test prod(ρ) ≈ runcircuit(prod(ρ0),compilecircuit(ρ0,gates,noise="DEP",p=0.1); apply_dag=true)
  reset_warn_order!()
  
  ## Mixed state, noisy circuit
  ρ0 = qubits(N,mixed=true)
  ρ = runcircuit(ρ0,gates;noise="DEP",p=0.1)
  @test prod(ρ) ≈ runcircuit(prod(ρ0),compilecircuit(ρ0,gates,noise="DEP",p=0.1); apply_dag=true)
  reset_warn_order!()

end

@testset "runcircuit: inverted gate order" begin
  N = 8
  gates = randomquantumcircuit(N,2)
  
  for n in 1:10
    s1 = rand(2:N)
    s2 = s1-1
    push!(gates,("CX", (s1,s2)))
  end
  ψ0 = qubits(N)
  ψ = runcircuit(ψ0, gates)
  @test prod(ψ) ≈ runcircuit(prod(ψ0),compilecircuit(ψ0,gates))

  U0 = circuit(N)
  U = runcircuit(U0,gates,get_unitary=true)
  disable_warn_order!()
  @test prod(U) ≈ runcircuit(prod(U0),compilecircuit(U0,gates))
end

@testset "runcircuit: long range gates" begin
  N = 8
  gates = randomquantumcircuit(N,2)
  
  for n in 1:10
    s1 = rand(1:N)
    s2 = rand(1:N)
    while s2 == s1
      s2 = rand(1:N)
    end
    push!(gates,("CX", (s1,s2)))
  end
  ψ0 = qubits(N)
  ψ = runcircuit(ψ0,gates)
  @test prod(ψ) ≈ runcircuit(prod(ψ0),compilecircuit(ψ0,gates)) 
  
  U0 = circuit(N)
  U = runcircuit(U0,gates,get_unitary=true)
  disable_warn_order!()
  @test prod(U) ≈ runcircuit(prod(U0),compilecircuit(U0,gates))
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
  ψ0 = qubits(N)
  gates = randomquantumcircuit(N,depth)
  ψ = runcircuit(ψ0,gates)
  ψ_vec = fullvector(ψ)
  prob = abs2.(ψ_vec)
  
  nshots = 100000
  samples = measure(ψ,nshots)
  @test size(samples)[1] == nshots
  @test size(samples)[2] == N
  data_prob = empiricalprobability(samples)
  @test prob ≈ data_prob atol=1e-2
end

@testset "measurement projections" begin
  N = 8
  nshots = 20
  ψ0 = qubits(N)
  bases = generatemeasurementsettings(N,nshots)
  
  depth = 8
  gates = randomquantumcircuit(N,depth)
  ψ = runcircuit(ψ0,gates)
  s = siteinds(ψ)

  for n in 1:nshots
    basis = bases[n,:]
    meas_gates = makemeasurementgates(basis)
    #meas_tensors = compilecircuit(ψ,meas_gates)
    ψ_out = runcircuit(ψ,meas_gates)
    x1 = measure(ψ_out,1)
    x1 .+= 1 
    
    if (basis[1] == "Z")
      ψ1 = ψ_out[1] * setelt(s[1]=>x1[1])
    else
      rotation = makegate(ψ_out,"meas$(basis[1])",1)
      ψ_r = ψ_out[1] * rotation
      ψ1 = noprime!(ψ_r) * setelt(s[1]=>x1[1])
    end
    for j in 2:N-1
      if (basis[j] == "Z")
        ψ1 = ψ1 * ψ_out[j] * setelt(s[j]=>x1[j])
      else
        rotation = makegate(ψ_out,"meas$(basis[j])",j)
        ψ_r = ψ_out[j] * rotation
        ψ1 = ψ1 * noprime!(ψ_r) * setelt(s[j]=>x1[j])
      end
    end
    if (basis[N] == "Z")
      ψ1 = (ψ1 * ψ_out[N] * setelt(s[N]=>x1[N]))[]
    else
      rotation = makegate(ψ_out,"meas$(basis[N])",N)
      ψ_r = ψ_out[N] * rotation
      ψ1 = (ψ1 * noprime!(ψ_r) * setelt(s[N]=>x1[N]))[]
    end
    
    # Change format of data
    x2 = []
    for j in 1:N
      if basis[j] == "X"
        if x1[j] == 1
          push!(x2,"projX+")
        else
          push!(x2,"projX-")
        end
      elseif basis[j] == "Y"
        if x1[j] == 1
          push!(x2,"projY+")
        else
          push!(x2,"projY-")
        end
      elseif basis[j] == "Z"
        if x1[j] == 1
          push!(x2,"projZ+")
        else
          push!(x2,"projZ-")
        end
      end
    end
  
    ψ2 = ψ_out[1] * dag(proj(x2[1],s[1]))
    for j in 2:N
      ψ_r = ψ_out[j] * dag(proj(x2[j],s[j]))
      ψ2 = ψ2 * ψ_r
    end
    ψ2 = ψ2[]
    @test ψ1 ≈ ψ2

    
    if (basis[1] == "Z")
      ψ1 = dag(ψ_out[1]) * setelt(s[1]=>x1[1])
    else
      rotation = makegate(ψ_out,"meas$(basis[1])",1)
      ψ_r = dag(ψ_out[1]) * dag(rotation)
      ψ1 = noprime!(ψ_r) * setelt(s[1]=>x1[1])
    end
    for j in 2:N-1
      if (basis[j] == "Z")
        ψ1 = ψ1 * dag(ψ_out[j]) * setelt(s[j]=>x1[j])
      else
        rotation = makegate(ψ_out,"meas$(basis[j])",j)
        ψ_r = dag(ψ_out[j]) * dag(rotation)
        ψ1 = ψ1 * noprime!(ψ_r) * setelt(s[j]=>x1[j])
      end
    end
    if (basis[N] == "Z")
      ψ1 = (ψ1 * dag(ψ_out[N]) * setelt(s[N]=>x1[N]))[]
    else
      rotation = makegate(ψ_out,"meas$(basis[N])",N)
      ψ_r = dag(ψ_out[N]) * dag(rotation)
      ψ1 = (ψ1 * noprime!(ψ_r) * setelt(s[N]=>x1[N]))[]
    end
  
    ψ2 = dag(ψ_out[1]) * proj(x2[1],s[1])
    for j in 2:N
      ψ_r = dag(ψ_out[j]) * proj(x2[j],s[j])
      ψ2 = ψ2 * ψ_r
    end
    ψ2 = ψ2[]
    @test ψ1 ≈ ψ2

  end
end

