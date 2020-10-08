
struct Choi{MT <: Union{MPO, LPDO}}
  M::MT
end

Base.length(Λ::Choi) = length(Λ.M)

Base.copy(Λ::Choi) = Choi(copy(Λ.M))

function Base.getindex(Λ::Choi, args...)
  error("getindex(Λ::Choi, args...) is purposefully not implemented yet.")
end

function Base.setindex!(Λ::Choi, args...)
  error("setindex!(Λ::Choi, args...) is purposefully not implemented yet.")
end

#
# Linear algebra/distance measures
#

tr(Λ::Choi; kwargs...) = tr(Λ.M; kwargs...)

fidelity_bound(Λ1::Choi, Λ2::Choi) =
  fidelity_bound(Λ1.M, Λ2.M)

function HDF5.write(parent::Union{HDF5File,HDF5Group},
                    name::AbstractString,
                    Λ::Choi)
  g = g_create(parent, name)
  attrs(g)["type"] = String(Symbol(typeof(Λ)))
  write(g, "M", Λ.M)
end

function HDF5.read(parent::Union{HDF5File, HDF5Group},
                   name::AbstractString,
                   ::Type{Choi{MT}}) where {MT}
  g = g_open(parent, name)
  M = read(g, "M", MT)
  return Choi(M)
end

