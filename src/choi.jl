
struct Choi{MPOT <: Union{MPO, LPDO}}
  M::MPOT
end

Base.length(C::Choi) = length(C.M)

Base.copy(C::Choi) = Choi(copy(C.M))

function Base.getindex(C::Choi, args...)
  error("getindex(C::Choi, args...) is purposefully not implemented yet.")
end

function Base.setindex!(C::Choi, args...)
  error("setindex!(C::Choi, args...) is purposefully not implemented yet.")
end


function makeUnitary(C::Choi{LPDO{MPS}})
  ψ = C.M.X
  U = MPO(ITensor[copy(ψ[j]) for j in 1:length(ψ)])
  prime!(U,tags="Output")
  removetags!(U, "Input")
  removetags!(U, "Output")
  return U
end

function makeChoi(U0::MPO)
  M = MPS(ITensor[copy(U0[j]) for j in 1:length(U0)])
  addtags!(M, "Input", plev = 0, tags = "Qubit")
  addtags!(M, "Output", plev = 1, tags = "Qubit")
  noprime!(M)
  return Choi(LPDO(M,ts""))
end

function LinearAlgebra.normalize!(C::Choi{LPDO{MPO}}; sqrt_localnorms! = [])
  normalize!(C.M; sqrt_localnorms! = sqrt_localnorms!)
  return C
end

function LinearAlgebra.normalize!(C::Choi{LPDO{MPS}}; sqrt_localnorms! = [])
  normalize!(C.M.X; localnorms! = sqrt_localnorms!)
  return C
end

function HDF5.write(parent::Union{HDF5File,HDF5Group},
                    name::AbstractString,
                    C::Choi)
  g = g_create(parent,name)
  attrs(g)["type"] = "Choi"
  if C.M isa MPO
    M = C.M
  elseif C.M isa LPDO
    M = C.M.X
  end
  attrs(g)["version"] = 1
  N = length(M)
  write(g, "rlim", M.rlim)
  write(g, "llim", M.llim)
  write(g, "length", N)
  for n=1:N
    write(g,"MPO[$(n)]", M[n])
  end
end

function HDF5.read(parent::Union{HDF5File,HDF5Group},
                   name::AbstractString,
                   ::Type{Choi})
  g = g_open(parent,name)
  if read(attrs(g)["type"]) != "Choi"
    error("HDF5 group or file does not contain MPO data")
  end
  N = read(g, "length")
  rlim = read(g, "rlim")
  llim = read(g, "llim")
  v = [read(g,"MPO[$(i)]",ITensor) for i in 1:N]
  return Choi(MPO( v, llim, rlim))
end
