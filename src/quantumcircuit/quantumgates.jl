# Identity
function gate_Id(i::Index)
  return itensor([1 0;
                  0 1],i',i)
end

function gate_X(i::Index)
  return itensor([0 1;
                  1 0],i',i)
end

function gate_Y(i::Index)
  return itensor([0 -im;
                  im  0],i',i)
end

function gate_Z(i::Index)
  return itensor([1  0;
                  0 -1],i',i)
end

function gate_H(i::Index)
  return (1/sqrt(2.))*itensor([1  1;
                               1 -1],i',i)
end

function gate_S(i::Index)
  return itensor([1  0;
                  0 im],i',i)
end

function gate_T(i::Index)
  return itensor([1  0;
                  0 exp(im*π/4)],i',i)
end

function gate_Kp(i::Index)
  return (1/sqrt(2.))*itensor([1   1;
                               im -im],i',i)
end

function gate_Km(i::Index)
  return (1/sqrt(2.))*itensor([1 -im;
                               1 im],i',i)
end

function gate_Rx(i::Index,θ::Float64)
  gate = [cos(θ/2.)     -im*sin(θ/2.);
          -im*sin(θ/2.)     cos(θ/2.)]
  return itensor(gate,i',i)
end

function gate_Ry(i::Index,θ::Float64)
  gate = [cos(θ/2.)     -sin(θ/2.);
          sin(θ/2.)     cos(θ/2.)]
  return itensor(gate,i',i)
end

function gate_Rz(i::Index,ϕ::Float64)
  gate = [exp(-im*ϕ/2.)  0;
          0              exp(im*ϕ/2.)]
  return itensor(gate,i',i)
end

function gate_Rn(i::Index,θ::Float64,ϕ::Float64,λ::Float64)
  gate = [cos(θ/2.)                -exp(im*λ) * sin(θ/2.);
          exp(im*ϕ) * sin(θ/2.)    exp(im*(ϕ+λ)) * cos(θ/2.)]
  return itensor(gate,i',i)
end

function gate_Sw(i::Index,j::Index)
  gate = reshape([1 0 0 0;
                  0 0 1 0;
                  0 1 0 0;
                  0 0 0 1],(2,2,2,2))
  return itensor(gate,i',j',i,j)
end

function gate_Cx(i::Index,j::Index)
  gate = reshape([1 0 0 0;
                  0 0 0 1;
                  0 0 1 0;
                  0 1 0 0],(2,2,2,2))
  return itensor(gate,i',j',i,j)
end

function gate_Cy(i::Index,j::Index)
  gate = reshape([1 0 0 0;
                  0 0 0 -im;
                  0 0 1 0;
                  0 im 0 0],(2,2,2,2))
  return itensor(gate,i',j',i,j)
end

function gate_Cz(i::Index,j::Index)
  gate = reshape([1 0 0 0;
                  0 1 0 0;
                  0 0 1 0;
                  0 0 0 -1],(2,2,2,2))
  return itensor(gate,i',j',i,j)
end

""" This is a comment 
"""
function gate(gate_id::String,site_ind::Index...;angles=nothing)
  if gate_id == "I"
    return gate_Id(site_ind[1])
  
  elseif gate_id == "X"
    return gate_X(site_ind[1])

  elseif gate_id == "Y"
    return gate_Y(site_ind[1])

  elseif gate_id == "Z"
    return gate_Z(site_ind[1])

  elseif gate_id == "H"
    return gate_H(site_ind[1])

  elseif gate_id == "S"
    return gate_S(site_ind[1])

  elseif gate_id == "T"
    return gate_T(site_ind[1])

  elseif gate_id == "Kp"
    return gate_Kp(site_ind[1])
  
  elseif gate_id == "Km"
    return gate_Km(site_ind[1])
  
  elseif gate_id == "Rx"
    return gate_Rx(site_ind[1],angles[1])
 
  elseif gate_id == "Ry"
    return gate_Ry(site_ind[1],angles[1])
  
  elseif gate_id == "Rz"
    return gate_Rz(site_ind[1],angles[1])
  
  elseif gate_id == "Rn"
    return gate_Rn(site_ind[1],angles[1],angles[2],angles[3])
  
  elseif gate_id == "Sw"
    return gate_Sw(site_ind[1],site_ind[2])
    
  elseif gate_id == "Cx"
    return gate_Cx(site_ind[1],site_ind[2])
    
  elseif gate_id == "Cy"
    return gate_Cy(site_ind[1],site_ind[2])
    
  elseif gate_id == "Cz"
    return gate_Cz(site_ind[1],site_ind[2])
    
  else
    throw(ArgumentError("Gate name '$gate_id' not recognized"))
  end

end
