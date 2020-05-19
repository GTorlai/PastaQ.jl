# Quantum circuit
function InitializeCircuit(N::Int)
  sites = [Index(2; tags="Site, n=$s") for s in 1:N]
  U = MPO(sites, "Id")
  return U
end

function InitializeQubits(N::Int)
  if (N==1)
    site = Index(2,tags="Site, n=1")
    m = itensor([1. 0.],site)
    psi = MPS([m])
  else
    sites = [Index(2; tags="Site, n=$s") for s in 1:N]
    psi = productMPS(sites, [1 for i in 1:length(sites)])
  end
  return psi
end

function ApplyGate!(M::Union{MPS,MPO},
                   gate_id::String,
                   site;
                   angles=nothing)
  if typeof(site) == Int 
    site_ind = firstind(M[site],"Site")
    gate = quantumgate(gate_id,site_ind,angles=angles)
    ApplyOneSiteGate!(M,gate,site)
  end
end
                
function ApplyOneSiteGate!(M::MPS,gate::ITensor,site::Int)
  M[site] = gate * M[site]
end

