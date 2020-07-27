
# TODO: rename findfirstsiteind and findfirstsiteinds
# to findsite.
# TODO: accept a keyword argument sitedict that
# is a dictionary from the site indices to the site.
"""
    findsite(M::Union{MPS, MPO}, is)

Return the site of the MPS or MPO that has the
site index or collection of site indices `is`.

# Examples
```julia
s = siteinds("S=1/2", 5)
ψ = randomMPS(s)
findsite(ψ, s[3]) == 3

M = MPO(s)
findsite(M, s[4]) == 4
findsite(M, s[4]') == 4
findsite(M, (s[4]', s[4])) == 4
isnothing(findsite(M, (s[4]', s[3])))
```
"""
findsite(ψ::ITensors.AbstractMPS, is) = findfirst(hasinds(is), ψ)

findsite(ψ::ITensors.AbstractMPS, s::Index) = findsite(ψ, IndexSet(s))


# TODO: make this the definition for siteind
"""
    firstsiteind(M::MPO, j::Int; plev = 0, kwargs...)

    siteind(M::MPO, j::Int; plev = 0, kwargs...)

Return the first site Index found of the MPS or MPO.

You can choose different filters, like prime level
and tags, with the `kwargs`.
"""
function firstsiteind(M::Union{MPS, MPO}, j::Int;
                      kwargs...)
  N = length(M)
  (N==1) && return firstind(M[1]; kwargs...)
  if j == 1
    si = uniqueind(M[j], M[j+1]; kwargs...)
  elseif j == N
    si = uniqueind(M[j], M[j-1]; kwargs...)
  else
    si = uniqueind(M[j], M[j-1], M[j+1]; kwargs...)
  end
  return si
end

ITensors.siteind(M::Union{MPS, MPO}, j::Int; kwargs...) =
  firstsiteind(M, j; kwargs...)

"""
    siteinds(M::MPO, j::Int; kwargs...)

Return the site Indices found of the MPO or MPO
at the site `j`.
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

"""
    siteinds(M::MPO; kwargs...)

Get a Vector of IndexSets the all of the site indices of M.
"""
ITensors.siteinds(M::MPO; kwargs...) =
  [siteinds(M, j; kwargs...) for j in 1:length(M)]

"""
    firstsiteinds(M::Union{MPS, MPO}; kwargs...)

Get a Vector of the first site Index found on each site of M.
"""
firstsiteinds(M::Union{MPS, MPO}; kwargs...) =
  [firstsiteind(M, j; kwargs...) for j in 1:length(M)]

"""
    findsites(ψ::Union{MPS, MPO},
              is::Vector)

Get a vector of the sites that contain the
indices `is`.

# Example
```julia
s = siteinds("S=1/2", 5)
ψ = randomMPS(s)
findsites(ψ, [s[3], s[2]]) == [3, 2]
```
"""
findsites(ψ::Union{MPS, MPO}, is::Vector) =
  [findsite(ψ, i) for i in is]

function findsites(ψ::Union{MPS, MPO},
                   o::ITensor)
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
  ψ = copy(ψ)
  if ortho == "left"
    orthocenter = b+1
  elseif ortho == "right"
    orthocenter = b
  end
  if ITensors.leftlim(ψ) < b - 1
    orthogonalize!(ψ, b)
  elseif ITensors.rightlim(ψ) > b + 2
    orthogonalize!(ψ, b + 1)
  end
  ψ[b:b+1, orthocenter = orthocenter, perm = [2, 1], kwargs...] = ψ[b] * ψ[b+1]
  return ψ
end

_number_inds(s::Index) = 1
_number_inds(s::IndexSet) = length(s)
_number_inds(sites) = sum(_num_inds(s) for s in sites)

function itensor_to_mps_or_mpo(MPST::Type{<:ITensors.AbstractMPS},
                               A::ITensor, sites;
                               firstlinkind::Union{Nothing, Index} = nothing,
                               lastlinkind::Union{Nothing, Index} = nothing,
                               orthocenter::Int = length(sites),
                               kwargs...)
  N = length(sites)
  @assert order(A) == _num_inds(sites) + !isnothing(firstlinkind) + !isnothing(lastlinkind)
  @assert hasinds(A, sites)
  if MPST <: MPO
    @assert hasinds(A, prime.(sites))
  end
  @assert isnothing(firstlinkind) || hasind(A, firstlinkind)
  @assert isnothing(lastlinkind) || hasind(A, lastlinkind)

  # TODO: To minimize work, loop from
  # 1:orthocenter and reverse(orthocenter:N)
  @assert 1 ≤ orthocenter ≤ N

  ψ = Vector{ITensor}(undef, N)
  Ã = A
  l = firstlinkind
  for n in 1:N-1
    s = sites[n]
    if MPST <: MPS
      Lis = isnothing(l) ? (s,) : (l, s)
    elseif MPST <: MPO
      Lis = isnothing(l) ? (s, s') : (l, s, s')
    end
    L, R = factorize(Ã, Lis; kwargs...)
    l = commonind(L, R)
    ψ[n] = L
    Ã = R
  end
  ψ[N] = Ã
  M = MPST(ψ)
  ITensors.setleftlim!(M, orthocenter-1)
  ITensors.setrightlim!(M, orthocenter+1)
  return M
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

# TODO: add a version that determines the sites
# from common site indices of ψ and A
"""
    setindex!(ψ::Union{MPS, MPO},
              A::ITensor,
              r::UnitRange{Int};
              orthocenter::Int = last(r),
              perm = nothing,
              kwargs...)

    replacesites!([...])

    replacesites([...])

Replace the sites in the range `r` with tensors made
from decomposing `A` into an MPS or MPO.

The MPS or MPO must be orthogonalized such that
```
firstsite ≤ ITensors.orthocenter(ψ) ≤ lastsite
```

Choose the new orthogonality center with `orthocenter`, which
should be within `r`.

Optionally, permute the order of the sites with `perm`.
"""
function Base.setindex!(ψ::MPST,
                        A::ITensor,
                        r::UnitRange{Int};
                        orthocenter::Int = last(r),
                        perm = nothing,
                        kwargs...) where {MPST <: Union{MPS, MPO}}
  # Replace the sites of ITensor ψ
  # with the tensor A, splitting up A
  # into MPS tensors
  firstsite = first(r)
  lastsite = last(r)
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

Base.setindex!(ψ::MPST,
               A::ITensor,
               r::UnitRange{Int},
               args::Pair{Symbol}...;
               kwargs...) where
                            {MPST <: Union{MPS, MPO}} =
  setindex!(ψ, A, r; args..., kwargs...)

replacesites!(ψ::ITensors.AbstractMPS, args...; kwargs...) =
  setindex!(ψ, args...; kwargs...)

replacesites(ψ::ITensors.AbstractMPS, args...; kwargs...) =
  setindex!(copy(ψ), args...; kwargs...)

# This version adds QN support
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

