# Adds QN support to MPO constructor
"""
    MPO([::Type{ElT} = Float64}, ]sites, ops::Vector{String})

Make an MPO with pairs of sites `s[i]` and `s[i]'`
and operators `ops` on each site.
"""
function ITensors.MPO(::Type{ElT},
             sites::Vector{<:Index},
             ops::Vector{String}) where {ElT <: Number}
  N = length(sites)
  its = Vector{ITensor}(undef, N)
  links = Vector{Index}(undef, N)
  for ii ∈ eachindex(sites)
    si = sites[ii]
    d = dim(si)
    spin_op = op(sites, ops[ii], ii)
    if hasqns(si)
      links[ii] = Index([QN() => 1], "Link,n=$ii")
    else
      links[ii] = Index(1, "Link,n=$ii")
    end
    local this_it
    if ii == 1
      this_it = emptyITensor(ElT, links[ii], si', dag(si))
      for jj in 1:d, jjp in 1:d
        this_it[links[ii](1), si[jj], si'[jjp]] = spin_op[si[jj], si'[jjp]]
      end
    elseif ii == N
      this_it = emptyITensor(ElT, dag(links[ii-1]), si', dag(si))
      for jj in 1:d, jjp in 1:d
        this_it[links[ii-1](1), si[jj], si'[jjp]] = spin_op[si[jj], si'[jjp]]
      end
    else
      this_it = emptyITensor(ElT, dag(links[ii-1]), links[ii], si', dag(si))
      for jj in 1:d, jjp in 1:d
        this_it[links[ii-1](1),
                links[ii](1),
                si[jj],
                si'[jjp]] = spin_op[si[jj], si'[jjp]]
      end
    end
    its[ii] = this_it
  end
  MPO(its)
end

