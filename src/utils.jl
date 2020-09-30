"""
    writesamples(data::Matrix, model::Union{MPS, MPO, LPDO, Choi},
              output_path::String)

Save data and model on file:

# Arguments:
  - `data`: array of measurement data
  - `model`: MPS, MPO, or Choi
  - `output_path`: path to file
"""
function writesamples(data::Matrix,
                   model::Union{MPS, MPO, LPDO, Choi},
                   output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout,"data",data)
    write(fout,"model",model)
  end
end

function writesamples(data::Matrix{<: Pair},
                   model::Union{MPS, MPO, LPDO, Choi},
                   output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "data_first", first.(data))
    write(fout, "data_last", last.(data))
    write(fout, "model", model)
  end
end

"""
    readsamples(input_path::String)

Load data and model from file:

# Arguments:
  - `input_path`: path to file
"""
function readsamples(input_path::String)
  fin = h5open(input_path, "r")
  g = g_open(fin, "model")
  typestring = read(attrs(g)["type"])
  modeltype = eval(Meta.parse(typestring))
  model = read(fin, "model", modeltype)
  if exists(fin, "data")
    data = read(fin, "data")
  elseif exists(fin, "data_first") && exists(fin, "data_last")
    data = read(fin, "data_first") .=> read(fin, "data_last")
  else
    error("File must contain either \"data\" for quantum state tomography data or \"data_first\" and \"data_second\" for quantum process tomography.")
  end
  close(fin)
  return data, model
end

"""
    PastaQ.fullvector(M::MPS; reverse::Bool = true)

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
    PastaQ.fullmatrix(M::MPO; reverse::Bool = true)
    PastaQ.fullmatrix(L::LPDO; reverse::Bool = true)

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

