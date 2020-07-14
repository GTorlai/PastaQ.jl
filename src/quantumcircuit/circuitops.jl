function makegate(gate_id::String, site_ind::Index; kwargs...)
  gate = quantumgate(gate_id, site_ind; kwargs...)
  return gate 
end

function makegate(M::MPS, gate_id::String, site::Int; kwargs...)
  site_ind = siteind(M,site)
  gate = quantumgate(gate_id, site_ind; kwargs...)
  return gate 
end

function makegate(M::MPO, gate_id::String, site::Int; kwargs...)
  site_ind = firstind(M[site], tags="Site", plev = 0)#siteind(M,site)
  gate = quantumgate(gate_id, site_ind; kwargs...)
  return gate 
end
function makegate(M::MPS,gate_id::String, site::Array; kwargs...)
  site_ind1 = siteind(M,site[1])
  site_ind2 = siteind(M,site[2])
  gate = quantumgate(gate_id,site_ind1,site_ind2; kwargs...)
  return gate
end

function makegate(M::MPO,gate_id::String, site::Array; kwargs...)
  site_ind1 = firstind(M[site[1]], tags="Site", plev = 0)
  site_ind2 = firstind(M[site[2]], tags="Site", plev = 0)
  gate = quantumgate(gate_id,site_ind1,site_ind2; kwargs...)
  return gate
end

function makegate(M::Union{MPS,MPO}, gatedata::NamedTuple)
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

function applygate!(M::MPO,
                   gate_id::String,
                   site::Int;
                   cutoff = 1e-10,
                   kwargs...)
  gate = makegate(M,gate_id,site; kwargs...)
  M[site] = gate * prime(M[site],tags="Site",plev=1)
  prime!(M[site],tags="Site",-1)
end

# Apply 1Q gate using a pre-generated gate tensor
function applygate!(M::MPS,gate::ITensor{2}; kwargs...)
  site = getsitenumber(firstind(gate,"Site")) 
  M[site] = gate * M[site]
  noprime!(M[site])
end

# Apply 1Q gate using a pre-generated gate tensor
function applygate!(M::MPO,gate::ITensor{2}; kwargs...)
  site = getsitenumber(firstind(gate; tags="Site", plev=0)) 
  M[site] = gate * prime(M[site],tags="Site",plev=1)
  prime!(M[site],tags="Site",-1)
end

function swap!(M::MPS,site1::Int,site2::Int,cutoff::Float64)
  nswaps = abs(site1-site2)-1
  if site1 > site2
    start = site2
  else
    start = site1
  end
  for n in 1:nswaps
    replacebond!(M,start+n-1,M[start+n-1]*M[start+n],swapsites=true,cutoff=cutoff)
  end
  newsite1 = start+nswaps
  newsite2 = newsite1 + 1
  return M,newsite1,newsite2 
end

function unswap!(M::MPS,site1::Int,site2::Int,cutoff::Float64)
  nswaps = abs(site1-site2)-1
  if site1 > site2
    start = site2
  else
    start = site1
  end
  for n in 1:nswaps
    replacebond!(M,start+nswaps-n,M[start+nswaps-n+1]*M[start+nswaps-n],swapsites=true,cutoff=cutoff)
  end
  return M
end

# Apply 2Q gate using a pre-generated gate tensor
function applygate!(M::MPS, gate::ITensor{4}; cutoff = 1e-10)
  s1 = getsitenumber(inds(gate,plev=1)[1]) 
  s2 = getsitenumber(inds(gate,plev=1)[2]) 
  
  if abs(s1-s2) != 1
    M,site1,site2 = swap!(M,s1,s2,cutoff)
  else
    site1=s1
    site2=s2
  end
  orthogonalize!(M,site1)
  blob = M[site1] * M[site2]
  blob = gate * blob
  noprime!(blob)
  U,S,V = svd(blob,inds(M[site1]),cutoff=cutoff)
  M[site1] = U
  M[site2] = S*V
  
  if abs(s1-s2) != 1
    M = unswap!(M,s1,s2,cutoff)
  end
end

# Apply 2Q gate using a pre-generated gate tensor
function applygate!(M::MPO, gate::ITensor{4}; cutoff = 1e-10)
  s1 = getsitenumber(inds(gate,plev=1)[1]) 
  s2 = getsitenumber(inds(gate,plev=1)[2]) 
  
  @assert(abs(s1-s2)==1)
  #TODO use swaps to handle long-range gates
 
  orthogonalize!(M,s1)
  blob = M[s1] * M[s2]
  blob = gate * prime(blob,tags="Site",plev=1)
  prime!(blob,tags="Site",-1)
  
  U,S,V = svd(blob,inds(M[s1]),cutoff=cutoff)
  M[s1] = U
  M[s2] = S*V
end

# Apply 2Q gate in the form ("Cx", [1,2])
function applygate!(M::Union{MPS,MPO},
                   gate_id::String,
                   site::Array;
                   cutoff = 1e-10,
                   kwargs...)
  
  # Construct the gate tensor
  gate = makegate(M,gate_id,site; kwargs...)
  applygate!(M,gate,cutoff=cutoff) 
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

