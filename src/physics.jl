function trace_mpo(M::MPO)
  N = length(M)
  L = M[1] * delta(dag(siteinds(M)[1]))
  if (N==1)
    return L
  end
  for j in 2:N
    trM = M[j] * delta(dag(siteinds(M)[j]))
    L = L * trM
  end
  return L[]
end

function groundstate(H::MPO; kwargs...)

  iter::Int64 = get(kwargs,:iter,20)
  cutoff::Float64 = get(kwargs,:cutoff,1E-10)
  energy::Bool = get(kwargs,:energy,false)
  energy_tol::Float64 = get(kwargs,:energy_tol,1E-5)

  sites = firstsiteinds(H)
  ψ0 = randomMPS(sites)
  sweeps = Sweeps(iter)
  maxdim!(sweeps, 10,10,20,30,40,50,60,80,100,150,200,300,400,500)
  cutoff!(sweeps, cutoff)
  noise!(sweeps,1E-10)
  observer = DMRGObserver(["Sz"],siteinds(ψ0),
                          energy_tol = energy_tol,
                          minsweeps = 10)
  E , ψ = dmrg(H,ψ0,sweeps,observer = observer,outputlevel=1)
  return (energy ? (E,ψ) : ψ)
end

"""
Transverse field Ising model
"""

function transversefieldising(N::Int64,h::Float64; kwargs...)
  
  bonds = [[j,j+1] for j in 1:N-1]
  return transversefieldising(N,bonds,h;kwargs...)
end

function transversefieldising(Lx::Int64,Ly::Int64,h::Float64;kwargs...)

  N = Lx * Ly
  lattice = square_lattice(Lx,Ly,yperiodic=false)
  bonds = [[b.s1,b.s2] for b in lattice]
  return transversefieldising(N,bonds,h;kwargs...)
end

function transversefieldising(N::Int64,
                              bonds::Array,
                              h::Float64;
                              kwargs...)
  
  β::Float64 = get(kwargs,:β,-1.0)
  hamiltonian::Bool = get(kwargs,:hamiltonian,false)

  # Ground state
  if β<0
    iter::Int64 = get(kwargs,:iter,20)
    cutoff::Float64 = get(kwargs,:cutoff,1E-10)
    sites = siteinds("S=1/2",N)
    ampo = AutoMPO()
    for b in bonds
      ampo .+=(-2.0,"Sz",b[1],"Sz",b[2])
    end
    for j in 1:N
      ampo += (-h,"Sx",j)
    end
    H = MPO(ampo,sites)
    ψ = groundstate(H,iter=iter,cutoff=cutoff)
    return (hamiltonian ? (H,ψ) : ψ) 
  
  # Finite temperature
  else
    τ::Float64 = get(kwargs,:τ,0.1)
    depth = β ÷ τ
    ρ0 = circuit(N)
    orthogonalize!(ρ0,1)
    gate_tensors = ITensor[]
    for d in 1:depth
      for b in bonds;
        z1 = gate(ρ0,"Z",b[1])
        z2 = gate(ρ0,"Z",b[2])
        zz = z1 * z2
        g  = exp(τ * zz)
        push!(gate_tensors,g)
      end
      for j in 1:N
        x = gate(ρ0,"X",j)
        g = exp(τ * h * x)
        push!(gate_tensors,g)
      end
    end
    ρ = runcircuit(ρ0,gate_tensors,state_evolution=false)
    
    # Normalize
    # TODO: substitute ITensor function here
    Z = trace_mpo(ρ)
    for j in 1:N
      ρ[j] = ρ[j] / Z^(1.0/N)
    end
    return (hamiltonian ? (H,ρ) : ρ) 
  end
end

