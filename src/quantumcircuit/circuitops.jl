
function gate(M::MPS, gatename::String, site::Int; kwargs...)
  site_ind = siteind(M,site)
  return gate(gatename, site_ind; kwargs...)
end

function gate(M::MPO, gatename::String, site::Int; kwargs...)
  site_ind = firstind(M[site], tags="Site", plev = 0)#siteind(M,site)
  return gate(gatename, site_ind; kwargs...)
end
function gate(M::MPS,gatename::String, site::Tuple; kwargs...)
  site_ind1 = siteind(M,site[1])
  site_ind2 = siteind(M,site[2])
  return gate(gatename,site_ind1,site_ind2; kwargs...)
end

function gate(M::MPO,gatename::String, site::Tuple; kwargs...)
  site_ind1 = firstind(M[site[1]], tags="Site", plev = 0)
  site_ind2 = firstind(M[site[2]], tags="Site", plev = 0)
  return gate(gatename,site_ind1,site_ind2; kwargs...)
end

function gate(M::Union{MPS,MPO}, gatedata::Tuple)
  return gate(M,gatedata...)
end

gate(M::Union{MPS,MPO}, gatename::String, sites::Union{Int, Tuple}, params::NamedTuple) =
  gate(M, gatename, sites; params...)

function applygate!(M::Union{MPS,MPO},gatename::String,sites::Union{Int,Tuple};kwargs...)
  g = gate(M,gatename,sites;kwargs...)
  Mc = apply(g,M;kwargs...)
  M[:] = Mc
  return M
end

function applygate!(M::Union{MPS,MPO},gate_tensor::ITensor; kwargs...)
  Mc = apply(gate_tensor,M;kwargs...)
  M[:] = Mc
  return M
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

