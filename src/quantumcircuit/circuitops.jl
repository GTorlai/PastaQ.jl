function makegate(gatename::String, site_ind::Index; kwargs...)
  return gate(gatename, site_ind; kwargs...)
end

function makegate(M::MPS, gatename::String, site::Int; kwargs...)
  site_ind = siteind(M,site)
  return gate(gatename, site_ind; kwargs...)
end

function makegate(M::MPO, gatename::String, site::Int; kwargs...)
  site_ind = firstind(M[site], tags="Site", plev = 0)#siteind(M,site)
  return gate(gatename, site_ind; kwargs...)
end
function makegate(M::MPS,gatename::String, site::Tuple; kwargs...)
  site_ind1 = siteind(M,site[1])
  site_ind2 = siteind(M,site[2])
  return gate(gatename,site_ind1,site_ind2; kwargs...)
end

function makegate(M::MPO,gatename::String, site::Tuple; kwargs...)
  site_ind1 = firstind(M[site[1]], tags="Site", plev = 0)
  site_ind2 = firstind(M[site[2]], tags="Site", plev = 0)
  return gate(gatename,site_ind1,site_ind2; kwargs...)
end

function makegate(M::Union{MPS,MPO}, gatedata::Tuple)
  return makegate(M,gatedata...)
end

makegate(M::Union{MPS,MPO}, gatename::String, sites::Union{Int, Tuple}, params::NamedTuple) =
  makegate(M, gatename, sites; params...)


function makekraus(M::MPO,noisename::String,site::Int; kwargs...)
  site_ind = firstind(M[site], tags="Site", plev = 0)
  return noise(noisename,site_ind; kwargs...)
end

function makekraus(M::MPO,noisename::String,site::Tuple; kwargs...)
  site_ind1 = siteind(M,site[1])
  site_ind2 = siteind(M,site[2])
  noise1    = noise(noisename,site_ind1; kwargs...)
  noise2    = noise(noisename,site_ind2; kwargs...)
  return noise1*noise2
end



function applygate!(M::Union{MPS,MPO},gatename::String,sites::Union{Int,Tuple};kwargs...)
  g = makegate(M,gatename,sites;kwargs...)
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

