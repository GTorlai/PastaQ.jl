
# Initialize the circuit MPO
function initializecircuit(N::Int)
  sites = [Index(2; tags="Site, n=$s") for s in 1:N]
  U = MPO(sites, "Id")
  return U
end

# Initialize the qubit MPS to |000...00>
function initializequbits(N::Int)
  sites = [Index(2; tags="Site, n=$s") for s in 1:N]
  psi = productMPS(sites, [1 for i in 1:length(sites)])
  return psi
end

# Make a 1Q gate tensor from gate_id,site and params
function makegate(M::MPS, gate_id::String, site::Int; kwargs...)
  site_ind = siteind(M,site)#firstind(M[site],"Site")
  gate = quantumgate(gate_id, site_ind; kwargs...)
  return gate 
end

# Make a 2Q gate tensor from gate_id,site and params
function makegate(M::MPS,gate_id::String, site::Array; kwargs...)
  site_ind1 = siteind(M,site[1])#firstind(M[site[1]],"Site")
  site_ind2 = siteind(M,site[2])#firstind(M[site[2]],"Site")
  gate = quantumgate(gate_id,site_ind1,site_ind2; kwargs...)
  return gate
end

# Make a gate from gatedata (gate_id,site and params)
function makegate(M::MPS, gatedata::NamedTuple)
  id = gatedata.gate
  site = gatedata.site
  params = get(gatedata, :params, NamedTuple())
  gate = makegate(M,id,site;params...)
  return gate
end

# Build the tensor of the circuit
function makecircuit(M::MPS,gatesdata::Array)
  gates = []
  for g in 1:length(gatesdata)
    gate = makegate(M,gatesdata[g])
    push!(gates,gate)
  end
  return gates
end

# Apply 1Q gate in the form ("X",1), ("Rn",2,(pi,pi,pi/2))
function applygate!(M::MPS,
                   gate_id::String,
                   site::Int;
                   cutoff = 1e-10,
                   kwargs...)
  gate = makegate(M,gate_id,site; kwargs...)
  M[site] = gate * M[site]
  noprime!(M[site])
end

# Apply 2Q gate in the form ("Cx", (1,2))
function applygate!(M::MPS,
                   gate_id::String,
                   site::Array;
                   cutoff = 1e-10,
                   kwargs...)
  
  @assert(abs(site[1]-site[2])==1)

  gate = makegate(M,gate_id,site; kwargs...)
  
  orthogonalize!(M,site[1])

  blob = M[site[1]] * M[site[2]]
  blob = gate * blob
  noprime!(blob)
  
  if site[1]==1
    row_ind = firstind(blob,tags="n=$(site[1])")
    U,S,V = svd(blob,row_ind,cutoff=cutoff)
    M[site[1]] = U*S
    M[site[2]] = V
  elseif site[1] == length(M)-1
    row_ind = firstind(blob,tags="n=$(site[2])")
    U,S,V = svd(blob,row_ind,cutoff=cutoff)
    M[site[1]] = V
    M[site[2]] = U*S
  else
    row_ind = (commonind(M[site[1]],M[site[1]-1]),
               firstind(blob,tags="n=$(site[1])"))
    U,S,V = svd(blob,row_ind,cutoff=cutoff)
    M[site[1]] = U*S
    M[site[2]] = V
  end
end

function getsitenumber(i::Index)
  for n in 1:length(tags(i))
    str_n = String(tags(i)[n])
    if startswith(str_n, "n=")
      return parse(Int64, replace(str_n, "n="=>""))
    end
  end
  return nothing
end


# Apply 1Q gate using a pre-generated gate tensor
function applygate!(M::MPS,gate::ITensor{2}; cutoff = 1e-10)
  site = getsitenumber(firstind(gate,"Site")) 
  M[site] = gate * M[site]
  noprime!(M[site])
end
# Apply 2Q gate using a pre-generated gate tensor
function applygate!(M::MPS, gate::ITensor{4}; cutoff = 1e-10)
  s1 = getsitenumber(inds(gate,plev=1)[1]) 
  s2 = getsitenumber(inds(gate,plev=1)[2]) 
  
  @assert(abs(s1-s2)==1)
  #TODO use swaps to handle long-range gates
  
  orthogonalize!(M,s1)
  blob = M[s1] * M[s2]
  blob = gate * blob
  noprime!(blob)
  if s1==1
    row_ind = firstind(blob,tags="n=$s1")
    U,S,V = svd(blob,row_ind,cutoff=cutoff)
    M[s1] = U*S
    M[s2] = V
  elseif s1 == length(M)-1
    row_ind = firstind(blob,tags="n=$s2")
    U,S,V = svd(blob,row_ind,cutoff=cutoff)
    M[s1] = V
    M[s2] = U*S
  else
    row_ind = (commonind(M[s1],M[s1-1]),
               firstind(blob,tags="n=$s1"))
    U,S,V = svd(blob,row_ind,cutoff=cutoff)
    M[s1] = U*S
    M[s2] = V
  end
end

function runcircuit!(M::MPS,gates;cutoff=1e-10)
  for g in 1:length(gates)
    gate = gates[g]
    applygate!(M,gates[g],cutoff=cutoff)
  end
end

