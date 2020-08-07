#
# qubit site type
#

ITensors.space(::SiteType"qubit") = 2

ITensors.state(::SiteType"qubit", ::StateName"0") = 1

ITensors.state(::SiteType"qubit", ::StateName"1") = 2

const GateName = OpName

macro GateName_str(s)
  GateName{ITensors.SmallString(s)}
end

const ProjName = OpName

macro ProjName_str(s)
  ProjName{ITensors.SmallString(s)}
end

const NoiseName = OpName

macro NoiseName_str(s)
  NoiseName{ITensors.SmallString(s)}
end


#
# Gate definitions.
# Gate names must not start with "proj".
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
# Measurement projections.
# Projector names must start with "proj".
#

proj(::ProjName"projX+") =
  [1/sqrt(2)
   1/sqrt(2)]

proj(::ProjName"X+") =
  proj("projX+")

proj(::ProjName"projX-") =
  [ 1/sqrt(2)
   -1/sqrt(2)]

proj(::ProjName"X-") =
  proj("projX-")

proj(::ProjName"projY+") =
  [  1/sqrt(2)
   im/sqrt(2)]

proj(::ProjName"Y+") =
  proj("projY+")

proj(::ProjName"projY-") =
  [  1/sqrt(2)
   -im/sqrt(2)]

proj(::ProjName"Y-") =
  proj("projY-")

proj(::ProjName"projZ+") =
  [1
   0]

proj(::ProjName"Z+") =
  proj("projZ+")

proj(::ProjName"projZ-") =
  [0
   1]

proj(::ProjName"Z-") =
  proj("projZ-")


#
# Get an ITensor gate from a gate definition
#


gate(::GateName{gn}; kwargs...) where {gn} =
  error("A gate with the name \"$gn\" has not been implemented yet. You can define it by overloading `gate(::GateName\"$gn\") = [...]`.")

gate(s::String) = gate(GateName(s))

function gate(gn::GateName, s::Index...; kwargs...)
  rs = reverse(s)
  return itensor(gate(gn; kwargs...), prime.(rs)..., dag.(rs)...)
end

gate(gn::String, s::Index...; kwargs...) =
  gate(GateName(gn), s...; kwargs...)

gate(gn::String, s::Vector{<:Index}, ns::Int...; kwargs...) =
  gate(gn, s[[ns...]]...; kwargs...)


#
#
#noise(s::String) = noise(NoiseName(s))
#function noise(nn::NoiseName, s::Index...; kwargs...)
#  rs = reverse(s)
#  return itensor(noise(nn; kwargs...), rs...)
#end
#
# Get an ITensor projector from a projector definition
#

function proj(PN::ProjName{pn}; kwargs...) where {pn}
  error("A projector with the name \"$pn\" has not been implemented yet. You can define it by overloading `proj(::ProjName\"$pn\") = [...]`.")
end

# TODO: this automatically appends "proj" to the front of
# the projector name.
# function proj(PN::ProjName{pn}; kwargs...) where {pn}
#   if isproj(PN)
#     error("A projector with the name \"$pn\" has not been implemented yet. You can define it by overloading `proj(::ProjName\"$pn\") = [...]`.")
#   else
#     @show pn
#     ppn = vcat(ITensors.SmallString("proj"), pn)
#     return proj(ProjName(ppn); kwargs...)
#   end
# end

proj(s::String) = proj(ProjName(s))

function proj(pn::ProjName, s::Index...; kwargs...)
  rs = reverse(s)
  return itensor(proj(pn; kwargs...), rs...)
end

proj(pn::String, s::Index...; kwargs...) =
  proj(ProjName(pn), s...; kwargs...)

proj(pn::String, s::Vector{<:Index}, ns::Int...; kwargs...) =
  proj(pn, s[[ns...]]...; kwargs...)

#
# op overload for ITensors
#

function isproj(::OpName{gn}) where {gn}
  ElT = eltype(gn)
  if gn[1] == ElT('p') &&
     gn[2] == ElT('r') &&
     gn[3] == ElT('o') &&
     gn[4] == ElT('j')
     return true
  end 
  return false
end

function ITensors.op(gn::GateName, ::SiteType"qubit", s::Index...; kwargs...)
  isproj(gn) && return proj(gn, s...; kwargs...)
  return gate(gn, s...; kwargs...)
end


#
# Noise models 
#

noise(s::String) = noise(NoiseName(s))

function noise(nn::NoiseName, s::Index...;kwargs...)
  rs = reverse(s)
  arr = noise(nn; kwargs...)
  kraus = Index(size(arr, 3),tags="kraus")  
  return itensor(noise(nn; kwargs...),prime.(rs)..., dag.(rs)..., kraus)
end

noise(nn::String, s::Index...; kwargs...) =
  noise(NoiseName(nn), s...; kwargs...)

noise(nn::String, s::Vector{<:Index}, ns::Int...; kwargs...) =
  noise(nn, s[[ns...]]...; kwargs...)

function noise(::NoiseName"AD"; γ::Number)
  kraus = zeros(2,2,2)
  kraus[:,:,1] = [1 0
                  0 sqrt(1-γ)]
  kraus[:,:,2] = [0 sqrt(γ)
                  0 0]
  return kraus 
end

#noise(::NoiseName"AD"; γ::Number) =
#  noise("noiseAD";γ=γ)

