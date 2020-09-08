function loadtrainingdataQST(input_path::String)
  data_file = h5open(input_path,"r")
  data = read(data_file,"data")
  #data .= "state" .* data
  target = read(data_file,"psi",MPS)
  return data,target
end

"""
Convert a data point from (sample,basis) -> data
Ex: (0,1,0,0) (X,Z,Y,X) -> (X+,Z-,Y+,X+)
"""
function convertdata(datapoint::Array,basis::Array)
  newdata = []
  for j in 1:length(datapoint)
    if basis[j] == "X"
      if datapoint[j] == 0
        push!(newdata,"stateX+")
      else
        push!(newdata,"stateX-")
      end
    elseif basis[j] == "Y"
      if datapoint[j] == 0
        push!(newdata,"stateY+")
      else
        push!(newdata,"stateY-")
      end
    elseif basis[j] == "Z"
      if datapoint[j] == 0
        push!(newdata,"stateZ+")
      else
        push!(newdata,"stateZ-")
      end
    end
  end
  return newdata
end

function fullvector(M::MPS; reverse::Bool = true)
  s = siteinds(M)
  if reverse
    s = Base.reverse(s)
  end
  C = combiner(s...)
  Mvec = prod(M) * dag(C)
  return array(Mvec)
end

function fullmatrix(M::MPO; reverse::Bool = true)
  s = firstsiteinds(M; plev = 0)
  if reverse
    s = Base.reverse(s)
  end
  C = combiner(s...)
  Mmat = prod(M) * dag(C) * C'
  c = combinedind(C)
  return array(permute(Mmat, c', c))
end

