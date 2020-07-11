
# TODO: make this the definition for siteind
"""
    firstsiteind(M::MPO, j::Int; plev = 0, kwargs...)

    siteind(M::MPO, j::Int; plev = 0, kwargs...)

Return the first site Index found of the MPO.

By default, it returns the first site Index with prime
level of 0. You can choose different filters, like tags,
with the `kwargs`.
"""
function firstsiteind(M::Union{MPS, MPO}, j::Int; plev::Int = 0, kwargs...)
  N = length(M)
  (N==1) && return firstind(M[1]; plev = plev, kwargs...)

  if j == 1
    si = uniqueind(M[j], M[j+1]; plev = plev, kwargs...)
  elseif j == N
    si = uniqueind(M[j], M[j-1]; plev = plev, kwargs...)
  else
    si = uniqueind(M[j], M[j-1], M[j+1]; plev = plev, kwargs...)
  end
  return si
end

# TODO: make this alias?
#const firstsiteind = siteind

# TODO: rename to allsiteinds?
"""
    siteinds(M::MPO, j::Int; kwargs...)

Return the site Indices found of the MPO.
"""
function ITensors.siteinds(M::Union{MPS, MPO}, j::Int; kwargs...)
  N = length(M)
  (N==1) && return inds(M[1]; kwargs...)

  if j == 1
    si = uniqueinds(M[j], M[j+1]; kwargs...)
  elseif j == N
    si = uniqueinds(M[j], M[j-1]; kwargs...)
  else
    si = uniqueinds(M[j], M[j-1], M[j+1]; kwargs...)
  end
  return si
end

# TODO: rename to allsiteinds?
"""
    siteinds(M::MPO; kwargs...)

Get a Vector of the site indices of M.
"""
ITensors.siteinds(M::MPO; kwargs...) =
  [siteinds(M, j; kwargs...) for j in 1:length(M)]

"""
    firstsiteinds(M::Union{MPS, MPO}; kwargs...)

Get a Vector of the first site Index found on each site of M.
"""
firstsiteinds(M::Union{MPS, MPO}; kwargs...) =
  [firstsiteind(M, j; kwargs...) for j in 1:length(M)]

function findsiteinds(ψ::Union{MPS, MPO},
                      inds::Vector)
  return [findfirstsiteinds(ψ, inds[n]) for n in 1:length(ψ)]
end

function findcommonsiteinds(o::ITensor,
                            ψ::Union{MPS, MPO})
  ns = Int[]
  for (n, ψn) in enumerate(ψ)
    if hascommoninds(o, ψn)
      push!(ns, n)
    end
  end
  ns
end

"""
    movesite(::Union{MPS, MPO}, n1::Int, n2::Int)

Create a new MPS where the site at `n1` is moved to `n2`
"""
function movesite(ψ::Union{MPS, MPO},
                  n1::Int, n2::Int;
                  orthocenter::Int = n2,
                  kwargs...)
  n1 == n2 && return copy(ψ)
  ψ = orthogonalize(ψ, n2)
  r = n1:n2-1
  ortho = "left"
  if n1 > n2
    r = reverse(n2:n1-1)
    ortho = "right"
  end
  for n in r
    ψ = swapbondsites(ψ, n; ortho = ortho, kwargs...)
  end
  ψ = orthogonalize(ψ, orthocenter)
  return ψ
end

function movesite(ns::Vector{Int},
                  n1::Int, n2::Int)
  n1 == n2 && return copy(ns)
  r = n1:n2-1
  if n1 > n2
    r = reverse(n2:n1-1)
  end
  for n in r
    #ns = swapbondsites(ns, n)
    ns = replace(ns, n => n+1, n+1 => n)
  end
  return ns
end

# TODO: make a permutesites(::MPS/MPO, perm)
# function that that a permutation of the sites
# p(1:N) for N sites
function movesites(ψ::Union{MPS, MPO},
                   ns, ns′; kwargs...)
  ψ = copy(ψ)
  N = length(ns)
  @assert N == length(ns′)
  p = sortperm(ns′)
  ns = ns[p]
  ns′ = ns′[p]
  ns = collect(ns)
  for i in 1:N
    ψ = movesite(ψ, ns[i], ns′[i]; kwargs...)
    ns = movesite(ns, ns[i], ns′[i])
  end
  return ψ
end

"""
    swapbondsites(ψ::MPO, b::Int; kwargs...)

Swap the sites `b` and `b+1`.
"""
function ITensors.swapbondsites(ψ::MPO, b::Int; kwargs...)
  ortho = get(kwargs, :ortho, "right")
  if ortho == "left"
    orthocenter = b+1
  elseif ortho == "right"
    orthocenter = b
  end
  if ITensors.leftlim(ψ) < b - 1
    ψ = orthogonalize(ψ, b)
  elseif ITensors.rightlim(ψ) > b + 2
    ψ = orthogonalize(ψ, b + 1)
  end
  #kwargs = setindex(values(kwargs), [2, 1], :parm)
  ψ = replacesites(ψ, ψ[b] * ψ[b+1];
                   firstsite = b, lastsite = b+1,
                   orthocenter = orthocenter,
                   perm = [2, 1],
                   kwargs...)
  return ψ
end

"""
    MPS(::ITensor, sites)

Construct an MPS from an ITensor by decomposing it site
by site.
"""
function MPS(A::ITensor, sites;
             firstlinkind::Union{Nothing, Index} = nothing,
             lastlinkind::Union{Nothing, Index} = nothing,
             orthocenter::Int = length(sites),
             kwargs...)
  N = length(sites)
  @assert order(A) == N + !isnothing(firstlinkind) + !isnothing(lastlinkind)
  @assert hasinds(A, sites)
  @assert isnothing(firstlinkind) || hasind(A, firstlinkind)
  @assert isnothing(lastlinkind) || hasind(A, lastlinkind)

  # TODO: generalize to other orthocenters.
  # To minimize work, may need to loop from
  # 1:orthocenter and reverse(orthocenter:N)
  @assert orthocenter == N

  ψ = Vector{ITensor}(undef, N)
  Ã = A
  l = firstlinkind
  for n in 1:N-1
    s = sites[n]
    Lis = isnothing(l) ? (s,) : (l, s)
    L, R = factorize(Ã, Lis; kwargs...)
    l = commonind(L, R)
    ψ[n] = L
    Ã = R
  end
  ψ[N] = Ã
  return MPS(ψ)
end

"""
    MPO(::ITensor, sites)

Construct an MPO from an ITensor by decomposing it site
by site.
"""
function MPO(A::ITensor, sites;
             firstlinkind::Union{Nothing, Index} = nothing,
             lastlinkind::Union{Nothing, Index} = nothing,
             orthocenter::Int = length(sites),
             kwargs...)
  N = length(sites)
  @assert order(A) == 2*N + !isnothing(firstlinkind) + !isnothing(lastlinkind)
  @assert hasinds(A, sites)
  @assert hasinds(A, prime.(sites))
  @assert isnothing(firstlinkind) || hasind(A, firstlinkind)
  @assert isnothing(lastlinkind) || hasind(A, lastlinkind)

  # TODO: generalize to other orthocenters.
  # To minimize work, may need to loop from
  # 1:orthocenter and reverse(orthocenter:N)
  @assert orthocenter == N

  ψ = Vector{ITensor}(undef, N)
  Ã = A
  l = firstlinkind
  for n in 1:N-1
    s = sites[n]
    Lis = isnothing(l) ? (s, s') : (l, s, s')
    L, R = factorize(Ã, Lis; kwargs...)
    l = commonind(L, R)
    ψ[n] = L
    Ã = R
  end
  ψ[N] = Ã
  M = MPO(ψ)
  ITensors.setleftlim!(M, orthocenter-1)
  ITensors.setrightlim!(M, orthocenter+1)
  return M
end

"""
    setindex!(::Union{MPS, MPO}, ::Union{MPS, MPO},
              r::UnitRange{Int64})

Sets a contiguous range of MPS/MPO tensors
"""
function Base.setindex!(ψ::MPST, ϕ::MPST,
                        r::UnitRange{Int64}) where {MPST <: Union{MPS, MPO}}
  @assert length(r) == length(ϕ)
  # TODO: accept r::Union{AbstractRange{Int}, Vector{Int}}
  # if r isa AbstractRange
  #   @assert step(r) = 1
  # else
  #   all(==(1), diff(r))
  # end
  llim = ITensors.leftlim(ψ)
  rlim = ITensors.rightlim(ψ)
  for (j, n) in enumerate(r)
    ψ[n] = ϕ[j]
  end
  if llim + 1 ≥ r[1]
    ITensors.setleftlim!(ψ, ITensors.leftlim(ϕ) + r[1] - 1)
  end
  if rlim - 1 ≤ r[end]
    ITensors.setrightlim!(ψ, ITensors.rightlim(ϕ) + r[1] - 1)
  end
  return ψ
end

# TODO: add a version of replacesites that
# determines `firstsite` and `lastsite` from common
# site indices of ψ and A
"""
    replacesites(ψ::Union{MPS, MPO},
                 A::ITensor;
                 firstsite::Int,
                 lastsite::Int,
                 orthocenter::Int = firstsite,
                 perm = nothing,
                 kwargs...)

Replace the sites between `firstsite` and `lastsite` with tensors made
from decomposing `A` into an MPS.

The MPS must be orthogonalized such that
```
firstsite ≤ ITensors.orthocenter(ψ) ≤ lastsite
```

Choose the new orthogonality center with `orthocenter`, which
should be within `firstsite:lastsite`.

Optionally, permute the order of the sites with `perm`.
"""
function replacesites(ψ::MPST,
                      A::ITensor;
                      firstsite::Int,
                      lastsite::Int,
                      orthocenter::Int = lastsite,
                      perm = nothing,
                      kwargs...) where {MPST <: Union{MPS, MPO}}
  # Replace the sites of ITensor ψ
  # with the tensor A, splitting up A
  # into MPS tensors
  ψ = copy(ψ)

  @assert firstsite ≤ ITensors.orthocenter(ψ) ≤ lastsite
  @assert firstsite ≤ ITensors.leftlim(ψ) + 1
  @assert ITensors.rightlim(ψ) - 1 ≤ lastsite

  # TODO: allow orthocenter outside of this
  # range, and orthogonalize/truncate as needed
  @assert firstsite ≤ orthocenter ≤ lastsite

  # Check that A has the proper common
  # indices with ψ
  l = linkind(ψ, firstsite-1)
  r = linkind(ψ, lastsite)
  # TODO: replace with commonsiteinds(::MPST, ::ITensor)?
  sites = collect(Iterators.flatten([siteinds(ψ, j) for j in firstsite:lastsite]))
  indsA = filter(x -> !isnothing(x), [l, sites..., r])

  @assert hassameinds(A, indsA)

  # For MPO case, restrict to 0 prime level
  sites = filter(hasplev(0), sites)

  if !isnothing(perm)
    sites = sites[[perm...]]
  end

  ψA = MPST(A, sites; firstlinkind = l,
                      lastlinkind = r,
                      kwargs...)
  #@assert prod(ψA) ≈ A

  ψ[firstsite:lastsite] = ψA

  return ψ
end

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

