"""
    savedata(model::Union{MPS,MPO},
             data::Array,output_path::String)

Save data and model on file:

# Arguments:
  - `model`: MPS or MPO
  - `data`: array of measurement data
  - `output_path`: path to file
"""
function savedata(model::Union{MPS,MPO},
                  data::Array,output_path::String)
  h5rewrite(output_path) do fout
    write(fout,"data",data)
    write(fout,"model",model)
  end
end

"""
    savedata(model::Union{MPS,MPO},
             data::Array,output_path::String)

Save data and model on file:

# Arguments:
  - `model`: MPS or MPO
  - `data_in` : array of preparation states
  - `data_out`: array of measurement data
  - `output_path`: path to file
"""
function savedata(model::Union{MPS,MPO},
                  data_in::Array,data_out::Array,
                  output_path::String)
  h5rewrite(output_path) do fout
    write(fout,"data_in",data_in)
    write(fout,"data_out",data_out)
    write(fout,"model",model)
  end
end

"""
    loaddata(input_path::String;process::Bool=false)

Load data and model from file:

# Arguments:
  - `input_path`: path to file
  - `process`: if `true`, load input/output data 
"""

function loaddata(input_path::String;process::Bool=false)
  fin = h5open(input_path,"r")
  
  g = g_open(fin,"model")
  typestring = read(attrs(g)["type"])
  modeltype = eval(Meta.parse(typestring))

  model = read(fin,"model",modeltype)
  
  if process
    data_in = read(fin,"data_in")
    data_out = read(fin,"data_out")
    return model,data_in,data_out
  else
    data = read(fin,"data")
    return model,data
  end
  close(fout)
end

"""
    fullvector(M::MPS; reverse::Bool = true)

Extract the full vector from an MPS
"""
function fullvector(M::MPS; reverse::Bool = true)
  s = siteinds(M)
  if reverse
    s = Base.reverse(s)
  end
  C = combiner(s...)
  Mvec = prod(M) * dag(C)
  return array(Mvec)
end

"""
    fullmatrix(M::MPO; reverse::Bool = true)

Extract the full matrix from an MPO
"""
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

