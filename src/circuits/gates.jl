#
# Qubit site type
#

ITensors.space(::SiteType"Qubit") = 2

ITensors.state(::SiteType"Qubit", ::StateName"0") = 1

ITensors.state(::SiteType"Qubit", ::StateName"1") = 2

const GateName = OpName

macro GateName_str(s)
  GateName{ITensors.SmallString(s)}
end

#
# Gate definitions.
# Gate names must not start with "state".
#

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
  [exp(-im*ϕ/2)           0
   0            exp(im*ϕ/2)]

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

function gate(::GateName"randU", N::Int)
  Q, _ = NDTensors.qr_positive(randn(ComplexF64, N, N))
  return Q
end

#
# State preparation gates
#

# State preparation: |0> -> |+>
gate(::GateName"prepX+") =
  gate("H")

# State preparation: |0> -> |->
gate(::GateName"prepX-") =
  [ 1/sqrt(2) 1/sqrt(2)
   -1/sqrt(2) 1/sqrt(2)]

# State preparation: |0> -> |r>
gate(::GateName"prepY+") =
  [ 1/sqrt(2)   1/sqrt(2)
   im/sqrt(2) -im/sqrt(2)]

# State preparation: |0> -> |l>
gate(::GateName"prepY-") =
  [  1/sqrt(2)  1/sqrt(2)
   -im/sqrt(2) im/sqrt(2)]

# State preparation: |0> -> |0>
gate(::GateName"prepZ+") =
  gate("I")

# State preparation: |0> -> |1>
gate(::GateName"prepZ-") =
  gate("X")

#
# Measurement gates
#

# Measurement rotation: |sX> -> |sZ>
gate(::GateName"measX") =
  gate("H")

# Measurement rotation: |sY> -> |sZ>
gate(::GateName"measY") =
  [1/sqrt(2) -im/sqrt(2)
   1/sqrt(2)  im/sqrt(2)]

# Measurement rotation: |sZ> -> |sZ>
gate(::GateName"measZ") =
  gate("I")


#
# Measurement projections onto a state.
# State projector names must start with "state".
#

gate(::GateName"stateX+") =
  [1/sqrt(2)
   1/sqrt(2)]

gate(::GateName"stateX-") =
  [ 1/sqrt(2)
   -1/sqrt(2)]

gate(::GateName"stateY+") =
  [  1/sqrt(2)
   im/sqrt(2)]

gate(::GateName"stateY-") =
  [  1/sqrt(2)
   -im/sqrt(2)]

gate(::GateName"stateZ+") =
  [1
   0]

gate(::GateName"state0") =
  gate("stateZ+")

gate(::GateName"stateZ-") =
  [0
   1]

gate(::GateName"state1") =
  gate("stateZ-")

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

function gate(::GateName"PD"; γ::Number)
  kraus = zeros(2,2,2)
  kraus[:,:,1] = [1 0
                  0 sqrt(1-γ)]
  kraus[:,:,2] = [0 0
                  0 sqrt(γ)]
  return kraus 
end

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

gate(::GateName"noiseDEP"; kwargs...) =
  gate("DEP";kwargs...)

gate(::GateName"noiseAD"; kwargs...) =
  gate("AD";kwargs...)

gate(::GateName"noisePD"; kwargs...) =
  gate("PD";kwargs...)


#
# Get an ITensor gate from a gate definition
#


gate(::GateName{gn}; kwargs...) where {gn} =
  error("A gate with the name \"$gn\" has not been implemented yet. You can define it by overloading `gate(::GateName\"$gn\") = [...]`.")

gate(s::String) = gate(GateName(s))

# Version that accepts a dimension for the gate,
# for n-qubit gates
gate(gn::GateName, N::Int; kwargs...) =
  gate(gn; kwargs...)

function gate(gn::GateName, s::Index...; kwargs...)
  rs = reverse(s)
  g = gate(gn, dim(s); kwargs...) 
  if ndims(g) == 1
    return itensor(g, rs...)
  elseif ndims(g) == 2
    return itensor(g, prime.(rs)..., dag.(rs)...)
  elseif ndims(g) == 3
    kraus = Index(size(g, 3),tags="kraus")  
    return itensor(g, prime.(rs)..., dag.(rs)..., kraus)
  end
  error("Gate definitions must be either Vector{T} (for a state), Matrix{T} (for a gate) or Array{T,3} (for a noise model). For gate name $gn, gate size is $(size(g)).") 
end

gate(gn::String, s::Index...; kwargs...) =
  gate(GateName(gn), s...; kwargs...)

gate(gn::String, s::Vector{<:Index}, ns::Int...; kwargs...) =
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

