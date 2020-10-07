
struct Choi{MT <: Union{MPO, LPDO}}
  M::MT
end

Base.length(C::Choi) = length(C.M)

Base.copy(C::Choi) = Choi(copy(C.M))

function Base.getindex(C::Choi, args...)
  error("getindex(C::Choi, args...) is purposefully not implemented yet.")
end

function Base.setindex!(C::Choi, args...)
  error("setindex!(C::Choi, args...) is purposefully not implemented yet.")
end

function HDF5.write(parent::Union{HDF5File,HDF5Group},
                    name::AbstractString,
                    C::Choi)
  g = g_create(parent, name)
  attrs(g)["type"] = String(Symbol(typeof(C)))
  write(g, "M", C.M)
end

function HDF5.read(parent::Union{HDF5File, HDF5Group},
                   name::AbstractString,
                   ::Type{Choi{MT}}) where {MT}
  g = g_open(parent, name)
  M = read(g, "M", MT)
  return Choi(M)
end

