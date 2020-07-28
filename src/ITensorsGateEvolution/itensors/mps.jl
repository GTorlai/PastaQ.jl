#"""
#    swapbondsites(ψ::MPO, b::Int; kwargs...)
#
#Swap the sites `b` and `b+1`.
#"""
#function ITensors.swapbondsites(ψ::MPO, b::Int; kwargs...)
#  ortho = get(kwargs, :ortho, "right")
#  ψ = copy(ψ)
#  if ortho == "left"
#    orthocenter = b+1
#  elseif ortho == "right"
#    orthocenter = b
#  end
#  if ITensors.leftlim(ψ) < b - 1
#    orthogonalize!(ψ, b)
#  elseif ITensors.rightlim(ψ) > b + 2
#    orthogonalize!(ψ, b + 1)
#  end
#  ψ[b:b+1,
#    orthocenter = orthocenter,
#    perm = [2, 1], kwargs...] = ψ[b] * ψ[b+1]
#  return ψ
#end

#_number_inds(s::Index) = 1
#_number_inds(s::IndexSet) = length(s)
#_number_inds(sites) = sum(_number_inds(s) for s in sites)
#
#"""
#    MPS(::ITensor, sites)
#
#    MPO(::ITensor, sites)
#
#Construct an MPS/MPO from an ITensor by decomposing it site
#by site.
#"""
#function (::Type{MPST})(A::ITensor, sites;
#                        firstlinkind::Union{Nothing, Index} = nothing,
#                        lastlinkind::Union{Nothing, Index} = nothing,
#                        orthocenter::Int = length(sites),
#                        kwargs...) where {MPST <: ITensors.AbstractMPS}
#  N = length(sites)
#  @assert order(A) == _number_inds(sites) + !isnothing(firstlinkind) + !isnothing(lastlinkind)
#  for s in sites
#    @assert hasinds(A, s)
#  end
#  @assert isnothing(firstlinkind) || hasind(A, firstlinkind)
#  @assert isnothing(lastlinkind) || hasind(A, lastlinkind)
#
#  @assert 1 ≤ orthocenter ≤ N
#
#  ψ = Vector{ITensor}(undef, N)
#  Ã = A
#  l = firstlinkind
#  # TODO: To minimize work, loop from
#  # 1:orthocenter and reverse(orthocenter:N)
#  # so the orthogonality center is set correctly.
#  for n in 1:N-1
#    Lis = sites[n]
#    if !isnothing(l)
#      Lis = push(Lis, l)
#    end
#    L, R = factorize(Ã, Lis; kwargs..., ortho = "left")
#    l = commonind(L, R)
#    ψ[n] = L
#    Ã = R
#  end
#  ψ[N] = Ã
#  M = MPST(ψ)
#  ITensors.setleftlim!(M, N-1)
#  ITensors.setrightlim!(M, N+1)
#  orthogonalize!(M, orthocenter)
#  return M
#end

#"""
#    MPS(::ITensor, sites)
#
#Construct an MPS from an ITensor by decomposing it site
#by site.
#"""
#ITensors.MPS(A::ITensor, sites; kwargs...) =
#  itensor_to_mps_or_mpo(MPS, A, sites; kwargs...)  
#
#"""
#    MPO(::ITensor, sites)
#
#Construct an MPO from an ITensor by decomposing it site
#by site.
#"""
#ITensors.MPO(A::ITensor, sites; kwargs...) =
#  itensor_to_mps_or_mpo(MPO, A, sites; kwargs...)  

#"""
#    MPS(::ITensor, sites)
#
#Construct an MPS from an ITensor by decomposing it site
#by site.
#"""
#function ITensors.MPS(A::ITensor, sites;
#                      firstlinkind::Union{Nothing, Index} = nothing,
#                      lastlinkind::Union{Nothing, Index} = nothing,
#                      orthocenter::Int = length(sites),
#                      kwargs...)
#  N = length(sites)
#  @assert order(A) == N + !isnothing(firstlinkind) + !isnothing(lastlinkind)
#  @assert hasinds(A, sites)
#  @assert isnothing(firstlinkind) || hasind(A, firstlinkind)
#  @assert isnothing(lastlinkind) || hasind(A, lastlinkind)
#
#  # TODO: generalize to other orthocenters.
#  # To minimize work, may need to loop from
#  # 1:orthocenter and reverse(orthocenter:N)
#  @assert orthocenter == N
#
#  ψ = Vector{ITensor}(undef, N)
#  Ã = A
#  l = firstlinkind
#  for n in 1:N-1
#    s = sites[n]
#    Lis = isnothing(l) ? (s,) : (l, s)
#    L, R = factorize(Ã, Lis; kwargs...)
#    l = commonind(L, R)
#    ψ[n] = L
#    Ã = R
#  end
#  ψ[N] = Ã
#  return MPS(ψ)
#end
#
#"""
#    MPO(::ITensor, sites)
#
#Construct an MPO from an ITensor by decomposing it site
#by site.
#"""
#function ITensors.MPO(A::ITensor, sites;
#                      firstlinkind::Union{Nothing, Index} = nothing,
#                      lastlinkind::Union{Nothing, Index} = nothing,
#                      orthocenter::Int = length(sites),
#                      kwargs...)
#  N = length(sites)
#  @assert order(A) == 2*N + !isnothing(firstlinkind) + !isnothing(lastlinkind)
#  @assert hasinds(A, sites)
#  @assert hasinds(A, prime.(sites))
#  @assert isnothing(firstlinkind) || hasind(A, firstlinkind)
#  @assert isnothing(lastlinkind) || hasind(A, lastlinkind)
#
#  # TODO: generalize to other orthocenters.
#  # To minimize work, may need to loop from
#  # 1:orthocenter and reverse(orthocenter:N)
#  @assert orthocenter == N
#
#  ψ = Vector{ITensor}(undef, N)
#  Ã = A
#  l = firstlinkind
#  for n in 1:N-1
#    s = sites[n]
#    Lis = isnothing(l) ? (s, s') : (l, s, s')
#    L, R = factorize(Ã, Lis; kwargs...)
#    l = commonind(L, R)
#    ψ[n] = L
#    Ã = R
#  end
#  ψ[N] = Ã
#  M = MPO(ψ)
#  ITensors.setleftlim!(M, orthocenter-1)
#  ITensors.setrightlim!(M, orthocenter+1)
#  return M
#end

#"""
#    setindex!(::Union{MPS, MPO}, ::Union{MPS, MPO},
#              r::UnitRange{Int64})
#
#Sets a contiguous range of MPS/MPO tensors
#"""
#function Base.setindex!(ψ::MPST, ϕ::MPST,
#                        r::UnitRange{Int64}) where {MPST <: Union{MPS, MPO}}
#  @assert length(r) == length(ϕ)
#  # TODO: accept r::Union{AbstractRange{Int}, Vector{Int}}
#  # if r isa AbstractRange
#  #   @assert step(r) = 1
#  # else
#  #   all(==(1), diff(r))
#  # end
#  llim = ITensors.leftlim(ψ)
#  rlim = ITensors.rightlim(ψ)
#  for (j, n) in enumerate(r)
#    ψ[n] = ϕ[j]
#  end
#  if llim + 1 ≥ r[1]
#    ITensors.setleftlim!(ψ, ITensors.leftlim(ϕ) + r[1] - 1)
#  end
#  if rlim - 1 ≤ r[end]
#    ITensors.setrightlim!(ψ, ITensors.rightlim(ϕ) + r[1] - 1)
#  end
#  return ψ
#end

## TODO: add a version that determines the sites
## from common site indices of ψ and A
#"""
#    setindex!(ψ::Union{MPS, MPO},
#              A::ITensor,
#              r::UnitRange{Int};
#              orthocenter::Int = last(r),
#              perm = nothing,
#              kwargs...)
#
#    replacesites!([...])
#
#    replacesites([...])
#
#Replace the sites in the range `r` with tensors made
#from decomposing `A` into an MPS or MPO.
#
#The MPS or MPO must be orthogonalized such that
#```
#firstsite ≤ ITensors.orthocenter(ψ) ≤ lastsite
#```
#
#Choose the new orthogonality center with `orthocenter`, which
#should be within `r`.
#
#Optionally, permute the order of the sites with `perm`.
#"""
#function Base.setindex!(ψ::MPST,
#                        A::ITensor,
#                        r::UnitRange{Int};
#                        orthocenter::Int = last(r),
#                        perm = nothing,
#                        kwargs...) where {MPST <: Union{MPS, MPO}}
#  # Replace the sites of ITensor ψ
#  # with the tensor A, splitting up A
#  # into MPS tensors
#  firstsite = first(r)
#  lastsite = last(r)
#  @assert firstsite ≤ ITensors.orthocenter(ψ) ≤ lastsite
#  @assert firstsite ≤ ITensors.leftlim(ψ) + 1
#  @assert ITensors.rightlim(ψ) - 1 ≤ lastsite
#
#  # TODO: allow orthocenter outside of this
#  # range, and orthogonalize/truncate as needed
#  @assert firstsite ≤ orthocenter ≤ lastsite
#
#  # Check that A has the proper common
#  # indices with ψ
#  l = linkind(ψ, firstsite-1)
#  r = linkind(ψ, lastsite)
#
#  sites = [siteinds(ψ, j) for j in firstsite:lastsite]
#
#  #s = collect(Iterators.flatten(sites))
#  indsA = filter(x -> !isnothing(x),
#                 [l, Iterators.flatten(sites)..., r])
#
#  @assert hassameinds(A, indsA)
#
#  # For MPO case, restrict to 0 prime level
#  #sites = filter(hasplev(0), sites)
#
#  if !isnothing(perm)
#    sites = sites[[perm...]]
#  end
#
#  ψA = MPST(A, sites;
#            firstlinkind = l,
#            lastlinkind = r,
#            kwargs...)
#  #@assert prod(ψA) ≈ A
#
#  ψ[firstsite:lastsite] = ψA
#
#  return ψ
#end

#Base.setindex!(ψ::MPST,
#               A::ITensor,
#               r::UnitRange{Int},
#               args::Pair{Symbol}...;
#               kwargs...) where
#                            {MPST <: Union{MPS, MPO}} =
#  setindex!(ψ, A, r; args..., kwargs...)
#
#replacesites!(ψ::ITensors.AbstractMPS, args...; kwargs...) =
#  setindex!(ψ, args...; kwargs...)
#
#replacesites(ψ::ITensors.AbstractMPS, args...; kwargs...) =
#  setindex!(copy(ψ), args...; kwargs...)

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

