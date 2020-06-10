" STANDARD GATES "

# Identity
function gate_I(i::Index)
  return itensor([1.0 0.0;
                  0.0 1.0],i',i)
end

# Pauli X
function gate_X(i::Index)
  return itensor([0.0 1.0;
                  1.0 0.0],i',i)
end

# Pauli Y
function gate_Y(i::Index)
  return itensor([0.0+0.0im -1.0im;
                  1.0im      0.0+0.0im],i',i)
end

# Pauli Z
function gate_Z(i::Index)
  return itensor([1.0  0.0;
                  0.0 -1.0],i',i)
end

const inv_sqrt2 = 0.7071067811865475

# Hadamard
function gate_H(i::Index)
  return itensor([inv_sqrt2  inv_sqrt2;
                  inv_sqrt2 -inv_sqrt2],i',i)
end

# S gate
function gate_S(i::Index)
  return itensor([1.0+0.0im 0.0im;
                  0.0im     1.0im],i',i)
end

# T gate
function gate_T(i::Index)
  return itensor([1.0+0.0im  0.0im;
                  0.0im      inv_sqrt2 + inv_sqrt2*im],i',i)
end

# Rotation around X axis
function gate_Rx(i::Index; θ::Float64)
  gate = [cos(θ/2.)     -im*sin(θ/2.);
          -im*sin(θ/2.)     cos(θ/2.)]
  return itensor(gate,i',i)
end

# Rotation around Y axis
function gate_Ry(i::Index; θ::Float64)
  gate = [cos(θ/2.)     -sin(θ/2.);
          sin(θ/2.)     cos(θ/2.)]
  return itensor(gate,i',i)
end

# Rotation around Z axis
function gate_Rz(i::Index; ϕ::Float64)
  gate = [exp(-im*ϕ/2.)  0;
          0              exp(im*ϕ/2.)]
  return itensor(gate,i',i)
end

# Rotation around generic axis
function gate_Rn(i::Index; θ::Float64,
                           ϕ::Float64,
                           λ::Float64)
  gate = [cos(θ/2.)                -exp(im*λ) * sin(θ/2.);
          exp(im*ϕ) * sin(θ/2.)    exp(im*(ϕ+λ)) * cos(θ/2.)]
  return itensor(gate,i',i)
end

# Swap gate
function gate_Sw(i::Index,j::Index)
  gate = reshape([1.0 0.0 0.0 0.0;
                  0.0 0.0 1.0 0.0;
                  0.0 1.0 0.0 0.0;
                  0.0 0.0 0.0 1.0],(2,2,2,2))
  return itensor(gate,i',j',i,j)
end

# Controlled-X
function gate_Cx(i::Index,j::Index)
  gate = reshape([1 0 0 0;
                  0 0 0 1;
                  0 0 1 0;
                  0 1 0 0],(2,2,2,2))
  return itensor(gate,i',j',i,j)
end

# Controlled-Y
function gate_Cy(i::Index,j::Index)
  gate = reshape([1 0 0 0;
                  0 0 0 -im;
                  0 0 1 0;
                  0 im 0 0],(2,2,2,2))
  return itensor(gate,i',j',i,j)
end

# Controlled-Z
function gate_Cz(i::Index,j::Index)
  gate = reshape([1 0 0 0;
                  0 1 0 0;
                  0 0 1 0;
                  0 0 0 -1],(2,2,2,2))
  return itensor(gate,i',j',i,j)
end

# State preparation: |0> -> |+>
function prep_Xp(i::Index)
  return gate_H(i)
end

# State preparation: |0> -> |->
function prep_Xm(i::Index)
  return itensor([ inv_sqrt2  inv_sqrt2;
                  -inv_sqrt2  inv_sqrt2],i',i)
end

# State preparation: |0> -> |r>
function prep_Yp(i::Index)
  return itensor([inv_sqrt2+0.0im   inv_sqrt2+0.0im;
                  inv_sqrt2*im     -inv_sqrt2*im],i',i)
end

# State preparation: |0> -> |l>
function prep_Ym(i::Index)
  return itensor([ inv_sqrt2+0.0im  inv_sqrt2+0.0im;
                  -inv_sqrt2*im     inv_sqrt2*im],i',i)
end

# State preparation: |0> -> |0>
function prep_Zp(i::Index)
  return gate_I(i)
end

# State preparation: |0> -> |1>
function prep_Zm(i::Index)
  return gate_X(i)
end

# Measurement rotation: |sX> -> |sZ>
function meas_X(i::Index)
  return gate_H(i)
end

# Measurement rotation: |sY> -> |sZ>
function meas_Y(i::Index)
  return itensor([inv_sqrt2+0.0im -inv_sqrt2*im;
                  inv_sqrt2+0.0im  inv_sqrt2*im],i',i)
end

# Measurement rotation: |sZ> -> |sZ>
function meas_Z(i::Index)
  return gate_I(i)
end

# A global dictionary of gate functions
quantumgates = Dict()

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
function quantumgate(gate_id::String,
                     site_inds::Index...;
                     kwargs...)
  return quantumgates[gate_id](site_inds...; kwargs...)
end
