"""
    makegate(M::MPS, gate_id::String, site::Int; kwargs...)

Create an ITensor for a single-qubit gate, using the indices 
of a given MPS. Additional parameters can be provided for parametric
gates.
# Example 1: X gate on site 3
```julia
psi = initializequbits(10)
gate = makegate(psi,"X",3)
```
# Example 2: ``\\phi = \\pi`` rotation around the Z axis on site 2
```julia
psi = initializequbits(10)
gate = makegate(psi, "Rz", 3, ϕ = π)
```
"""
function makegate(M::MPS, gate_id::String, site::Int; kwargs...)
  site_ind = siteind(M,site)
  gate = quantumgate(gate_id, site_ind; kwargs...)
  return gate 
end

"""
    makegate(M::MPS, gate_id::String, site::Array; kwargs...)

Create an ITensor for a two-qubit gate, using the indices 
of a given MPS. Additional parameters can be provided for parametric
gates.
# Example: Cx gate on site [2,3]
```julia
psi = initializequbits(10)
gate = makegate(psi,"Cx",[2,3])
```
"""
function makegate(M::MPS,gate_id::String, site::Array; kwargs...)
  site_ind1 = siteind(M,site[1])
  site_ind2 = siteind(M,site[2])
  gate = quantumgate(gate_id,site_ind1,site_ind2; kwargs...)
  return gate
end

"""
    makegate(M::MPS, gatedata::NamedTuple)

Create an ITensor for a two-qubit gate, using the indices 
of a given MPS. The gate is specified by a NamedTuple with structure
gatedata = (gate,site,params)

# Example: Rn rotation
```julia
psi = initializequbits(10)
gatedata = (gate = "Rn",
            site = 4,
            params = (θ = 1.7,ϕ = 0.9,λ = 4.2))

gate = makegate(psi,gatedata)
```
"""
function makegate(M::MPS, gatedata::NamedTuple)
  id = gatedata.gate
  site = gatedata.site
  params = get(gatedata, :params, NamedTuple())
  gate = makegate(M,id,site;params...)
  return gate
end

function applygate!(M::MPS,
                   gate_id::String,
                   site::Int;
                   cutoff = 1e-10,
                   kwargs...)
  gate = makegate(M,gate_id,site; kwargs...)
  M[site] = gate * M[site]
  noprime!(M[site])
end

# Apply 2Q gate in the form ("Cx", [1,2])
function applygate!(M::MPS,
                   gate_id::String,
                   site::Array;
                   cutoff = 1e-10,
                   kwargs...)
  
  # Check that the qubits are NN
  # TODO remove and insert swap gates
  @assert(abs(site[1]-site[2])==1)
  
  # Construct the gate tensor
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

# Apply 1Q gate using a pre-generated gate tensor
function applygate!(M::MPS,gate::ITensor{2}; kwargs...)
  site = getsitenumber(firstind(gate,"Site")) 
  M[site] = gate * M[site]
  noprime!(M[site])
end

# Apply 2Q gate using a pre-generated gate tensor
function applygate!(M::MPS, gate::ITensor{4}; cutoff = 1e-10)
  s1 = getsitenumber(inds(gate,plev=1)[1]) 
  s2 = getsitenumber(inds(gate,plev=1)[2]) 
  
  if abs(s1-s2)!=1
    #TODO  
    # do swaps
    nothing
  end
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

# Retrieve the qubit number from a site index
function getsitenumber(i::Index)
  for n in 1:length(tags(i))
    str_n = String(tags(i)[n])
    if startswith(str_n, "n=")
      return parse(Int64, replace(str_n, "n="=>""))
    end
  end
  return nothing
end

