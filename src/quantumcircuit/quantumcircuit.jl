# Quantum circuit
function initializecircuit(N::Int)
  sites = [Index(2; tags="Site, n=$s") for s in 1:N]
  U = MPO(sites, "Id")
  return U
end

function initializequbits(N::Int)
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

function applygate!(M::MPS,
                   gate_id::String,
                   site::Int;
                   kwargs...)
  site_ind = firstind(M[site],"Site")
  gate = quantumgate(gate_id, site_ind; kwargs...)
  M[site] = gate * M[site]
  noprime!(M[site])
end

function applygate!(M::MPS,
                   gate_id::String,
                   site::Array;
                   kwargs...)
  #site_ind = firstind(M[site],"Site")
  #gate = quantumgate(gate_id, site_ind; kwargs...)
  #M[site] = gate * M[site]
  #noprime!(M[site])
end
