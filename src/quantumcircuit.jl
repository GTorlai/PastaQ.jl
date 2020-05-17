# Quantum Circuit structure
struct QuantumCircuit
  N::Int
  seed::Int
  rng::MersenneTwister
  sites::Vector{Index}
  links::Vector{Index}
  U::MPO  
  gate_list::Vector{Any}
  infos::Dict
end

# Constructor
function QuantumCircuit(;N::Int,seed::Int=1234)
  # Random number generator
  rng = MersenneTwister(seed)

  # Site and link indices
  sites = [Index(2; tags="site, s=$s") for s in 1:N]
  links  = [Index(1; tags="link, l=$l") for l in 1:N-1]
  
  # Initialize the MPO
  u = ITensor[]
  push!(u,ITensor(reshape([1 0;0 1],(2,1,2)),sites[1],links[1],sites[1]'))
  for j in 2:N-1
    push!(u,ITensor(reshape([1 0;0 1],(2,1,2,1)),sites[j],links[j-1],sites[j]',links[j]))
  end
  push!(u,ITensor(reshape([1 0;0 1],(2,1,2)),sites[N],links[N-1],sites[N]'))
  U = MPO(u)
  orthogonalize!(U,1)
  gate_list = []
  infos = Dict()
  
  return QuantumCircuit(N,seed,rng,sites,links,U,gate_list,infos)
end

# Initialize all qubits in the |0>
function InitializeQubits(qc::QuantumCircuit)
  state = ITensor[]
  push!(state,ITensor(reshape([1 0],(2,1)),qc.sites[1],qc.links[1]))
  for j in 2:qc.N-1
    push!(state,ITensor(reshape([1 0],(1,2,1)),qc.links[j-1],qc.sites[j],qc.links[j]))
  end
  push!(state,ITensor(reshape([1 0],(1,2)),qc.links[qc.N-1],qc.sites[qc.N]))
  mps_state = MPS(state)
  #orthogonalize!(mps_state,1)
  return mps_state
end

# Prepare a product state of pauli eigenstates
function StatePreparation(qc::QuantumCircuit,qgates::QuantumGates,state_id::Vector{Int})
  mps_state = InitializeQubits(qc)
  for j in 1:qc.N
    if state_id[j] == 1      #|0>
      nothing
    elseif state_id[j] == 2  #|1>
      ApplySingleQubitGate!(mps_state,qgates.X,j)
    elseif state_id[j] == 3  #|+>
      ApplySingleQubitGate!(mps_state,qgates.H,j)
    elseif state_id[j] == 4  #|->
      ApplySingleQubitGate!(mps_state,qgates.X,j)
      ApplySingleQubitGate!(mps_state,qgates.H,j)
    elseif state_id[j] == 5  #|r>
      ApplySingleQubitGate!(mps_state,qgates.K,j)
    else state_id[j] == 6  #|l>
      ApplySingleQubitGate!(mps_state,qgates.X,j)
      ApplySingleQubitGate!(mps_state,qgates.K,j)
    end
  end
  return mps_state
end

# Apply single-qubit gate to the bra side of an MPO
function ApplySingleQubitGate!(mpo::MPO,gate::ITensor,site::Int)
  site_ind = inds(mpo[site],tags="site")[1]
  replaceinds!(gate,inds(gate),[site_ind'',site_ind])
  mpo[site] = gate * mpo[site]
  mpo[site] = setprime(mpo[site],tags="site",plev=2,0)
end

function RandomSingleQubitLayer!(qc::QuantumCircuit)
  for j in 1:qc.N
    angles = rand!(qc.rng, zeros(3))
    θ = π * angles[1]
    ϕ = 2π * angles[2]
    λ = 2π * angles[3]
    u3 = U3(θ,ϕ,λ)
    ApplySingleQubitGate!(qc.U,u3,j)
    push!(qc.gate_list,["u3",j,[θ,ϕ,λ]])
  end
end

function LoadQuantumCircuit(qc::QuantumCircuit,qgates::QuantumGates,gate_list)
  for g in 1:size(gate_list)[1]
    gate_id = gate_list[g,1]
    site = gate_list[g,2]
    if (size(gate_list[g,:])[1]==3)
      angles = gate_list[g,3]
    #  println("Gate ",gate_id," on site ",site," with angles ",angles)
    #else
    #  println("Gate ",gate_id," on site ",site)
    end
    if (gate_id == "U3")
      θ = angles[1]
      ϕ = angles[2]
      λ = angles[3]
      u3 = U3(θ,ϕ,λ)
      ApplySingleQubitGate!(qc.U,u3,site)
      push!(qc.gate_list,["U3",site,[θ,ϕ,λ]])
    elseif (gate_id == "H")
      ApplySingleQubitGate!(qc.U,qgates.H,site)
      push!(qc.gate_list,["H",site])
    end
  end
end

#function RunQuantumCircuit(qc::QuantumCircuit,qgates::QuantumGates;
#                           state_id::Vector{Int}=ones(Int,qc.N))
#  mps_state = StatePreparation(qc,qgates,state_id)
#  if isempty(qc.gate_list) error("Gate list is empty!") end
#  
#  for g in size(qc.gate_list)[1]
#    
#  end
#end
## Apply single-qubit gate to the bra side of an MPO
#function ApplySingleQubitGate!(M::AbstractMPS,gate::ITensor,site::Int)
#  site_ind = inds(M[site],tags="site")[1]
#  replaceinds!(gate,inds(gate),[site_ind'',site_ind])
#  M[site] = gate * M[site]
#  M[site] = setprime(M[site],tags="site",plev=2,0)
#end
#                              
## Applu single-qubit gate to an MPS
#function ApplySingleQubitGate!(mps::MPS,gate::ITensor,site::Int)
#  site_ind = inds(mps[site],tags="site")[1]
#  replaceinds!(gate,inds(gate),[site_ind',site_ind])
#  mps[site] = gate * mps[site]
#  mps[site] = setprime(mps[site],tags="site",plev=1,0)
#end

#function SingleQubitRandomLayer!(qc::QuantumCircuit)
#  for j in 1:qc.N
#    angles = rand!(qc.rng, zeros(3))
#    θ = π * angles[1]
#    ϕ = 2π * angles[2]
#    λ = 2π * angles[3]
#    u3 = U3(θ,ϕ,λ)
#    ApplySingleQubitGate!(qc,u3,j)
#    push!(qc.gate_list,["u3",j,[θ,ϕ,λ]])
#  end
#end
#
#function PopulateInfoDict!(qc::QuantumCircuit)
#  qc.infos["N"] = qc.N
#  qc.infos["seed"] = qc.seed
#end
# 
