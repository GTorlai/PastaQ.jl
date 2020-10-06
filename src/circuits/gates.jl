#
# Gate definitions.
# Gate names must not start with "basis".
#

const GateName = OpName

macro GateName_str(s)
  GateName{ITensors.SmallString(s)}
end

#
# State-like gates, used to define product input states
#

# TODO: add an arbitrary state specified by angles

inputstate(::StateName"X+") =
  [1/sqrt(2)
   1/sqrt(2)]

inputstate(::StateName"X-") =
  [ 1/sqrt(2)
   -1/sqrt(2)]

inputstate(::StateName"Y+") =
  [  1/sqrt(2)
   im/sqrt(2)]

inputstate(::StateName"Y-") =
  [  1/sqrt(2)
   -im/sqrt(2)]

inputstate(::StateName"Z+") =
  [1
   0]

inputstate(::StateName"0") =
  inputstate("Z+")

inputstate(::StateName"Z-") =
  [0
   1]

inputstate(::StateName"1") =
  inputstate("Z-")

inputstate(sn::String; kwargs...) = inputstate(StateName(sn); kwargs...)

inputstate(sn::String, i::Index; kwargs...) =
  itensor(inputstate(sn; kwargs...), i)

#
# 1-qubit gates
#

gate(::GateName"Id") =
  [1 0
   0 1]

gate(::GateName"I") =
  gate("Id")

gate(::GateName"X") =
  [0 1
   1 0]

gate(::GateName"σx") =
  gate("X") 

gate(::GateName"σ1") =
  gate("X") 

gate(::GateName"√NOT") =
  [(1+im)/2 (1-im)/2
   (1-im)/2 (1+im)/2]

gate(::GateName"√X") =
  gate("√NOT")

gate(::GateName"Y") =
  [ 0 -im
   im   0]

gate(::GateName"σy") =
  gate("Y") 

gate(::GateName"σ2") =
  gate("Y") 

gate(::GateName"iY") =
  [ 0 1
   -1 0]

gate(::GateName"iσy") =
  gate("iY")

gate(::GateName"iσ2") =
  gate("iY")

# Rϕ with ϕ = π
gate(::GateName"Z") =
  [1  0
   0 -1]

gate(::GateName"σz") =
  gate("Z")

gate(::GateName"σ3") =
  gate("Z")

gate(::GateName"H") =
  [1/sqrt(2) 1/sqrt(2)
   1/sqrt(2) -1/sqrt(2)]

# Rϕ with ϕ = π/2
gate(::GateName"Phase") =
  [1  0
   0 im]

gate(::GateName"P") =
  gate("Phase")

gate(::GateName"S") =
  gate("Phase")

# Rϕ with ϕ = π/4
gate(::GateName"π/8") =
  [1  0
   0  1/sqrt(2) + im/sqrt(2)]

gate(::GateName"T") =
  gate("π/8")

# Rotation around X-axis
gate(::GateName"Rx"; θ::Number) =
  [    cos(θ/2)  -im*sin(θ/2)
   -im*sin(θ/2)      cos(θ/2)]

# Rotation around Y-axis
gate(::GateName"Ry"; θ::Number) =
  [cos(θ/2) -sin(θ/2)
   sin(θ/2)  cos(θ/2)]

# Rotation around Z-axis
gate(::GateName"Rz"; ϕ::Number) =
  [1         0
   0 exp(im*ϕ)]

# Rotation around generic axis n̂
gate(::GateName"Rn";
     θ::Real, ϕ::Real, λ::Real) =
  [          cos(θ/2)    -exp(im*λ)*sin(θ/2)
   exp(im*ϕ)*sin(θ/2) exp(im*(ϕ+λ))*cos(θ/2)]

gate(::GateName"Rn̂"; kwargs...) =
  gate("Rn"; kwargs...)

#
# 2-qubit gates
#

gate(::GateName"CNOT") =
  [1 0 0 0
   0 1 0 0
   0 0 0 1
   0 0 1 0]

gate(::GateName"CX") =
  gate("CNOT")

gate(::GateName"CY") =
  [1 0  0   0
   0 1  0   0
   0 0  0 -im
   0 0 im   0]

gate(::GateName"CZ") =
  [1 0 0  0
   0 1 0  0
   0 0 1  0
   0 0 0 -1]

# Same as CRn with (θ = 0, λ = 0)
gate(::GateName"CRz"; ϕ::Real) =
  [1 0 0         0
   0 1 0         0
   0 0 1         0
   0 0 0 exp(im*ϕ)]

gate(::GateName"CRn";
     θ::Real, ϕ::Real, λ::Real) =
  [1 0                 0                       0
   0 1                 0                       0
   0 0           cos(θ/2)    -exp(im*λ)*sin(θ/2)
   0 0 exp(im*ϕ)*sin(θ/2) exp(im*(ϕ+λ))*cos(θ/2)]

gate(::GateName"SWAP") =
  [1 0 0 0
   0 0 1 0
   0 1 0 0
   0 0 0 1]

gate(::GateName"Sw") =
  gate("SWAP")

gate(::GateName"√SWAP") =
  [1        0        0 0
   0 (1+im)/2 (1-im)/2 0
   0 (1-im)/2 (1+im)/2 0
   0        0        0 1]

# Ising (XX) coupling gate
gate(::GateName"XX"; ϕ::Number) =
  [    cos(ϕ)          0          0 -im*sin(ϕ)
            0     cos(ϕ) -im*sin(ϕ)          0
            0 -im*sin(ϕ)     cos(ϕ)          0
   -im*sin(ϕ)          0          0     cos(ϕ)]

# TODO: Ising (YY) coupling gate
#gate(::GateName"YY"; ϕ::Number) =
#  [...]

# TODO: Ising (ZZ) coupling gate
#gate(::GateName"ZZ"; ϕ::Number) =
#  [...]

#
# 3-qubit gates
#

gate(::GateName"Toffoli") =
  [1 0 0 0 0 0 0 0
   0 1 0 0 0 0 0 0
   0 0 1 0 0 0 0 0
   0 0 0 1 0 0 0 0
   0 0 0 0 1 0 0 0
   0 0 0 0 0 1 0 0
   0 0 0 0 0 0 0 1
   0 0 0 0 0 0 1 0]

gate(::GateName"CCNOT") =
  gate("Toffoli")

gate(::GateName"CCX") =
  gate("Toffoli")

gate(::GateName"TOFF") =
  gate("Toffoli")

gate(::GateName"Fredkin") =
  [1 0 0 0 0 0 0 0
   0 1 0 0 0 0 0 0
   0 0 1 0 0 0 0 0
   0 0 0 1 0 0 0 0
   0 0 0 0 1 0 0 0
   0 0 0 0 0 0 1 0
   0 0 0 0 0 1 0 0
   0 0 0 0 0 0 0 1]

gate(::GateName"CSWAP") =
  gate("Fredkin")

gate(::GateName"CS") =
  gate("Fredkin")

#
# 4-qubit gates
#

gate(::GateName"CCCNOT") =
  [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
   0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0
   0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0
   0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0
   0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0
   0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0
   0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0
   0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0
   0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
   0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0
   0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0
   0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0
   0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0
   0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0
   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
   0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0]

#
# n-qubit gates
#

function gate(::GateName"randU", N::Int = 2;
              eltype = ComplexF64,
              random_matrix = randn(eltype, N, N))
  Q, _ = NDTensors.qr_positive(random_matrix)
  return Q
end

#
# Noise model gate definitions
#

function gate(::GateName"AD"; γ::Number)
  kraus = zeros(2,2,2)
  kraus[:,:,1] = [1 0
                  0 sqrt(1-γ)]
  kraus[:,:,2] = [0 sqrt(γ)
                  0 0]
  return kraus 
end

# To accept the gate name "amplitude_damping"
gate(::GateName"amplitud"; kwargs...) = gate("AD"; kwargs...)

function gate(::GateName"PD"; γ::Number)
  kraus = zeros(2,2,2)
  kraus[:,:,1] = [1 0
                  0 sqrt(1-γ)]
  kraus[:,:,2] = [0 0
                  0 sqrt(γ)]
  return kraus 
end

# To accept the gate name "phase_damping"
gate(::GateName"phase_da"; kwargs...) = gate("PD"; kwargs...)

# To accept the gate name "dephasing"
gate(::GateName"dephasin"; kwargs...) = gate("PD"; kwargs...)

function gate(::GateName"DEP"; p::Number)
  kraus = zeros(Complex{Float64},2,2,4)
  kraus[:,:,1] = sqrt(1-p)   * [1 0 
                                0 1]
  kraus[:,:,2] = sqrt(p/3.0) * [0 1 
                                1 0]
  kraus[:,:,3] = sqrt(p/3.0) * [0 -im 
                                im 0]
  kraus[:,:,4] = sqrt(p/3.0) * [1  0 
                                0 -1]
  return kraus 
end

# To accept the gate name "depolarizing"
gate(::GateName"depolari"; kwargs...) = gate("DEP"; kwargs...)

gate(::GateName"noiseDEP"; kwargs...) =
  gate("DEP";kwargs...)

gate(::GateName"noiseAD"; kwargs...) =
  gate("AD";kwargs...)

gate(::GateName"noisePD"; kwargs...) =
  gate("PD";kwargs...)

#
# Qubit site type
#

import ITensors: space

space(::SiteType"Qubit") = 2

import ITensors: state

state(::SiteType"Qubit", ::StateName"0") = 1

state(::SiteType"Qubit", ::StateName"1") = 2

#
# Basis definitions (eigenbases of measurement gates)
#

function phase(v::AbstractVector{ElT}) where {ElT <: Number}
  for x in v
    absx = abs(x)
    if absx > 1e-3
      return x / abs(x)
    end
  end
  return one(ElT)
end

function eigenbasis(GN::GateName; dag::Bool = false, kwargs...)
  _, U = eigen(Hermitian(gate(GN; kwargs...)))
  # Sort eigenvalues largest to smallest (defaults to smallest to largest)
  U = reverse(U; dims = 2)
  # Fix the sign of the eigenvectors
  for n in 1:size(U, 2)
    v = @view U[:, n]
    p = phase(v)
    v ./= p
  end
  if dag
    return copy(U')
  end
  return U
end

#
# Get an ITensor gate from a gate definition
#

function Base.startswith(ss::ITensors.SmallString, st::String)
  for j in 1:length(st)
    if ss[j] ≠ st[j]
      return false
    end
  end
  return true
end

function gate(::GateName{gn}; kwargs...) where {gn}
  gn_st = String(gn)
  if startswith(gn_st, "basis")
    GN = GateName(replace(gn_st, "basis" => ""))
    return eigenbasis(GN; kwargs...)
  end
  error("A gate with the name \"$gn\" has not been implemented yet. You can define it by overloading `gate(::GateName\"$gn\") = [...]`.")
end

# Maybe use Base.invokelatest since certain gate overloads
# may be made on the fly with @eval
gate(s::String; kwargs...) = gate(GateName(s); kwargs...)

# Version that accepts a dimension for the gate,
# for n-qubit gates
gate(gn::GateName, N::Int; kwargs...) =
  gate(gn; kwargs...)

function gate(gn::GateName, s1::Index, ss::Index...; kwargs...)
  s = tuple(s1, ss...)
  rs = reverse(s)
  g = gate(gn, dim(s); kwargs...) 
  if ndims(g) == 1
    # TODO:
    #error("gate must have more than one dimension, use state(...) for state vectors.")
    return itensor(g, rs...)
  elseif ndims(g) == 2
    return itensor(g, prime.(rs)..., dag.(rs)...)
  elseif ndims(g) == 3
    kraus = Index(size(g, 3),tags="kraus")  
    return itensor(g, prime.(rs)..., dag.(rs)..., kraus)
  end
  error("Gate definitions must be either Vector{T} (for a state), Matrix{T} (for a gate) or Array{T,3} (for a noise model). For gate name $gn, gate size is $(size(g)).") 
end

function gate(gn::String, s::Index...; kwargs...)
  if length(gn) > 8
    gn = gn[1:8]
  end
  return gate(GateName(gn), s...; kwargs...)
end

gate(gn::String, s::Vector{<: Index}, ns::Int...; kwargs...) =
  gate(gn, s[[ns...]]...; kwargs...)

#
# op overload so that ITensor functions like MPO(::AutoMPO) can use the gate
# definitions of the "Qubit" site type
#

function ITensors.op(gn::GateName,
                     ::SiteType"Qubit",
                     s::Index...;
                     kwargs...)
  return gate(gn, s...; kwargs...)
end

