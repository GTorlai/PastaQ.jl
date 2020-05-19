# Quantum circuit

function InitializeCircuit(N::Int)
  sites = [Index(2; tags="Site, n=$s") for s in 1:N]
  U = MPO(sites, "Id")
  return U
end

function InitializeQubits(N::Int)
  if (N==1)
    site = Index(2,tags="Site, n=1")
    psi = itensor([1. 0.],site)
    #psi = MPS([m])
  else
    sites = [Index(2; tags="Site, n=$s") for s in 1:N]
    psi = productMPS(sites, [1 for i in 1:length(sites)])
  end
  return psi
end

#function ApplyGate(M::Union{MPS,MPO},
#                   gate_id::String,
#                   sites;
#                   angles=nothing)
#  if typeof(sites) == Int 
#    site_ind = siteind(M,sites)
#    gate = quantumgate(gate_id,site_ind,angles)
#    @show gate
#  end
#end
                

#function ApplyOneSiteGate!(M::Union{MPS,MPO},gate::ITensor,site::Int)
#  @show siteind(M[site])
#
#end

