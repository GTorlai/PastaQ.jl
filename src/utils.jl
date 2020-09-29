"""
    savedata(data::Array, model::Union{MPS,MPO,Choi},
             output_path::String)

Save data and model on file:

# Arguments:
  - `data`: array of measurement data
  - `model`: MPS, MPO, or Choi
  - `output_path`: path to file
"""
function savedata(data::Array,
                  model::Union{MPS,MPO,Choi},
                  output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout,"data",data)
    if model isa Choi
      model = model.M
    end
    write(fout, "model", model)
  end
end

"""
    savedata(data_in::Array,
             data_out::Array,
             model::Union{MPS,MPO,Choi},
             output_path::String)

Save data and model on file:

# Arguments:
  - `data_in` : array of preparation states
  - `data_out`: array of measurement data
  - `model`: MPS, MPO, or Choi
  - `output_path`: path to file
"""
function savedata(data_in::Array,
                  data_out::Array,
                  model::Union{MPS,MPO,Choi},
                  output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout,"data_in",data_in)
    write(fout,"data_out",data_out)
    if model isa Choi
      model = model.M
    end
    write(fout,"model",model)
  end
end

"""
    loaddata(input_path::String;process::Bool=false)

Load data and model from file:

# Arguments:
  - `input_path`: path to file
  - `process`: if `true`, data is treated as coming from measuring a process, and loads both input and output data
"""

function loaddata(input_path::String; process::Bool = false)
  fin = h5open(input_path,"r")
  
  g = g_open(fin,"model")
  typestring = read(attrs(g)["type"])
  modeltype = eval(Meta.parse(typestring))

  model = read(fin, "model", modeltype)
  
  if process
    data_in = read(fin,"data_in")
    data_out = read(fin,"data_out")
    close(fin)
    return data_in, data_out, model
  else
    data = read(fin,"data")
    close(fin)
    return data, model
  end
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

    fullmatrix(L::LPDO; reverse::Bool = true)

Extract the full matrix from an MPO or LPDO, returning a Julia Matrix.
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

fullmatrix(L::LPDO; kwargs...) = fullmatrix(MPO(L); kwargs...)

# TEMPORARY FUNCTION
# TODO: remove when `firstsiteinds(ψ::MPS)` is implemented
function hilbertspace(L::LPDO) 
  return  (L.X isa MPS ? siteinds(L.X) : firstsiteinds(L.X))
end

hilbertspace(M::Union{MPS,MPO}) = hilbertspace(LPDO(M))

