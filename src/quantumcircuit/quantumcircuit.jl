# Quantum circuit

function InitializeCircuitMPO(N::Int)
  sites = [Index(2; tags="Site, n=$s") for s in 1:N]
  U = MPO(sites, "Id")
  return U
end

function InitializeQubits(sites::IndexSet)
  psi = productMPS(sites, [1 for i in 1:length(sites)])
  return psi
end
