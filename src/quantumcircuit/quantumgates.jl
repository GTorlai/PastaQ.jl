# Identity
function gate_I(T = ComplexF64)
  return T[1 0;
           0 1]
end

# Pauli X
function gate_X(T = ComplexF64)
  return T[0 1;
           1 0]
end

# Pauli Y
function gate_Y(T = ComplexF64)
  return T[0+0im 0-im;
           0+im  0+0im]
end

# Pauli Z
function gate_Z(T = ComplexF64)
  return T[1  0;
           0 -1]
end

const inv_sqrt2 = 0.7071067811865475

# Hadamard
function gate_H(T = ComplexF64)
  return T[inv_sqrt2  inv_sqrt2;
           inv_sqrt2 -inv_sqrt2]
end

# S gate
function gate_S(T = ComplexF64)
  return T[1+0im 0+0im;
           0+0im 0+im]
end

# T gate
function gate_T(T = ComplexF64)
  return T[1.0+0.0im  0.0+0.0im;
           0.0+0.0im  inv_sqrt2+inv_sqrt2*im]
end

# Rotation around X axis
function gate_Rx(T = ComplexF64; θ::Float64)
  return T[cos(θ/2)+0.0im   0.0-im*sin(θ/2.);
           0.0-im*sin(θ/2.) cos(θ/2.)+0.0im]
end

# Rotation around Y axis
function gate_Ry(T = ComplexF64; θ::Float64)
  return T[cos(θ/2) -sin(θ/2);
           sin(θ/2)  cos(θ/2)]
end

# Rotation around Z axis
function gate_Rz(T = ComplexF64; ϕ::Real)
  return T[exp(-im*ϕ/2.) 0.0+0.0im;
           0.0+0.0im     exp(im*ϕ/2)]
end

# Rotation around generic axis
function gate_Rn(T = ComplexF64;
                 θ::Float64,
                 ϕ::Float64,
                 λ::Float64)
  return T[cos(θ/2)+0.0im       -exp(im*λ) * sin(θ/2);
           exp(im*ϕ) * sin(θ/2)  exp(im*(ϕ+λ)) * cos(θ/2)]
end

# Swap gate
function gate_Sw(T = ComplexF64)
  return T[1 0 0 0;
           0 0 1 0;
           0 1 0 0;
           0 0 0 1]
end

# Controlled-X
function gate_Cx(T = ComplexF64)
  return T[1 0 0 0;
           0 1 0 0;
           0 0 0 1;
           0 0 1 0]
  #return T[1 0 0 0;
  #         0 0 0 1;
  #         0 0 1 0;
  #         0 1 0 0]
end

# Controlled-Y
function gate_Cy(T = ComplexF64)
  #return T[1 0 0 0;
  #         0 0 0 -im;
  #         0 0 1 0;
  #         0 im 0 0]
  return T[1 0 0 0;
           0 1 0 0;
           0 0 0 -im;
           0 0 im 0]
end

# Controlled-Z
function gate_Cz(T = ComplexF64)
  return T[1 0 0 0;
           0 1 0 0;
           0 0 1 0;
           0 0 0 -1]
end

# State preparation: |0> -> |+>
function prep_Xp(T = ComplexF64)
  return gate_H(T)
end

# State preparation: |0> -> |->
function prep_Xm(T = ComplexF64)
  return T[ inv_sqrt2  inv_sqrt2;
           -inv_sqrt2  inv_sqrt2]
end

# State preparation: |0> -> |r>
function prep_Yp(T = ComplexF64)
  return T[inv_sqrt2+0.0im   inv_sqrt2+0.0im;
           0.0+inv_sqrt2*im  0.0-inv_sqrt2*im]
end

# State preparation: |0> -> |l>
function prep_Ym(T = ComplexF64)
  return T[ inv_sqrt2+0.0im  inv_sqrt2+0.0im;
           0.0-inv_sqrt2*im  0.0+inv_sqrt2*im]
end

# State preparation: |0> -> |0>
function prep_Zp(T = ComplexF64)
  return gate_I(T)
end

# State preparation: |0> -> |1>
function prep_Zm(T = ComplexF64)
  return gate_X(T)
end

# Measurement rotation: |sX> -> |sZ>
function meas_X(T = ComplexF64)
  return gate_H(T)
end

# Measurement rotation: |sY> -> |sZ>
function meas_Y(T = ComplexF64)
  return T[inv_sqrt2+0.0im 0.0-inv_sqrt2*im;
           inv_sqrt2+0.0im 0.0+inv_sqrt2*im]
end

# Measurement rotation: |sZ> -> |sZ>
function meas_Z(T = ComplexF64)
  return gate_I(T)
end

# A global dictionary of gate functions
quantumgates = Dict{String, Function}()

# Default gates
quantumgates["I"]  = gate_I
quantumgates["X"]  = gate_X
quantumgates["Y"]  = gate_Y
quantumgates["Z"]  = gate_Z
quantumgates["H"]  = gate_H
quantumgates["S"]  = gate_S
quantumgates["T"]  = gate_T
quantumgates["Rx"] = gate_Rx
quantumgates["Ry"] = gate_Ry
quantumgates["Rz"] = gate_Rz
quantumgates["Rn"] = gate_Rn
quantumgates["Sw"] = gate_Sw
quantumgates["Cx"] = gate_Cx
quantumgates["Cy"] = gate_Cy
quantumgates["Cz"] = gate_Cz

quantumgates["pX+"] = prep_Xp
quantumgates["pX-"] = prep_Xm
quantumgates["pY+"] = prep_Yp
quantumgates["pY-"] = prep_Ym
quantumgates["pZ+"] = prep_Zp
quantumgates["pZ-"] = prep_Zm

quantumgates["mX"] = meas_X
quantumgates["mY"] = meas_Y
quantumgates["mZ"] = meas_Z

"""
    quantumgate(gate_id::String,
                site_inds::Index...;
                kwargs...)

Make the specified gate with the specified indices.

# Example
```julia
i = Index(2; tags = "i")
quantumgate("X", i)
```
"""
function quantumgate(T,
                     gate_id::String,
                     site_inds::Index...;
                     reverse_order=true,
                     kwargs...)
  if reverse_order
    is = IndexSet(reverse(site_inds)...)
  else
    is = IndexSet(site_inds...)
  end
  return itensor(quantumgates[gate_id](T; kwargs...), is'..., is...)
end

function quantumgate(gate_id::String,
                     site_inds::Index...;
                     kwargs...)
  return quantumgate(ComplexF64, gate_id, site_inds...;
                     kwargs...)
end

measproj_Xp(T = ComplexF64) = T[inv_sqrt2; inv_sqrt2]
measproj_Xm(T = ComplexF64) = T[inv_sqrt2; -inv_sqrt2]
measproj_Yp(T = ComplexF64) = T[inv_sqrt2+0.0im; 0.0+im*inv_sqrt2]
measproj_Ym(T = ComplexF64) = T[inv_sqrt2+0.0im; 0.0-im*inv_sqrt2]
measproj_Zp(T = ComplexF64) = T[1; 0]
measproj_Zm(T = ComplexF64) = T[0; 1]

measprojections = Dict{String, Function}()
measprojections["X+"] = measproj_Xp
measprojections["X-"] = measproj_Xm
measprojections["Y+"] = measproj_Yp
measprojections["Y-"] = measproj_Ym
measprojections["Z+"] = measproj_Zp
measprojections["Z-"] = measproj_Zm

function measproj(T::Type{<:Number},
                  proj_id::String,
                  site_ind::Index)
  return itensor(measprojections[proj_id](T), site_ind)
end

function measproj(proj_id::String,
                  site_ind::Index)
  return measproj(ComplexF64, proj_id, site_ind)
end

