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

function QuantumCircuit(;N::Int,seed::Int=1234)
  rng = MersenneTwister(seed)
  sites = [Index(2; tags="site, s=$s") for s in 1:N]
  links  = [Index(1; tags="link, l=$l") for l in 1:N-1]
  
  u = ITensor[]
  # Site 1
  push!(u,ITensor(reshape([1 0;0 1],(2,1,2)),sites[1],links[1],sites[1]'))
  for j in 2:N-1
    push!(u,ITensor(reshape([1 0;0 1],(2,1,2,1)),sites[j],links[j-1],sites[1]',links[j]))
  end
  push!(u,ITensor(reshape([1 0;0 1],(2,1,2)),sites[N],links[N-1],sites[N]'))
  U = MPO(u)

  gate_list = []
  infos = Dict()
  
  return QuantumCircuit(N,seed,rng,sites,links,U,gate_list,infos)

end

function ApplySingleQubitGate!(qc::QuantumCircuit,gate::ITensor,site::Int)
  site_ind = inds(qc.U[site],tags="site")[1]
  replaceinds!(gate,inds(gate),[site_ind'',site_ind])
  qc.U[site] = gate * qc.U[site]
  qc.U[site] = setprime(qc.U[site],tags="site",plev=2,0)
end
                              
function SingleQubitRandomLayer!(qc::QuantumCircuit,qg::QuantumGates)
  for j in 1:qc.N
    angles = rand!(qc.rng, zeros(3))
    θ = π * angles[1]
    ϕ = 2π * angles[2]
    λ = 2π * angles[3]
    u3 = U3(θ,ϕ,λ)
    ApplySingleQubitGate!(qc,u3,j)
    push!(qc.gate_list,["u3",j,[θ,ϕ,λ]])
  end
end

function PopulateInfoDict(qc::QuantumCircuit)
  qc.infos["N"] = qc.N
  qc.infos["seed"] = qc.seed
end
 

