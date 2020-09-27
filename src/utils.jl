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
                  data::Array,
                  output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
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
                  data_in::Array,
                  data_out::Array,
                  output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
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



function hilbertspace(L::LPDO)
  return  (L.X isa MPS ? siteinds(L.X) : firstsiteinds(L.X))
end

hilbertspace(M::Union{MPS,MPO}) = hilbertspace(LPDO(M))

  
function replacehilbertspace!(A::LPDO,B::LPDO)
  M = A.X
  hA = hilbertspace(A)
  hB = hilbertspace(B)
  # Check if there are purification indices
  purification_tag = any(x -> hastags(x,"Purifier") , M)
  
  @assert length(A)==length(B)
  for j in 1:length(M)
    # Object to be modified is MPS
    if M isa MPS
      replaceind!(M[j],hA[j],hB[j])
    # Object be modified is MPO
    elseif M isa MPO
      # TODO make it general for tagsets
      # Object is MPO with purification indices
      if purification_tag
      #  # Object has rank-4 bulk tensors (it's state X, ρ=XX\dagger)
        if length(size(M[2]))==4
          replaceind!(M[j],hA[j],hB[j])
        # Object has rank-5 bulk tensors (it.s a process)
        elseif length(size(M[2]))==5
          replaceind!(M[j],siteinds(M)[j][1],siteinds(B.X)[j][1])
          replaceind!(M[j],siteinds(M)[j][2],siteinds(B.X)[j][2])
        end
      ## Object is a regular MPO
      else
        if B.X isa MPS
          replaceind!(M[j],siteinds(M)[j][1],hB[j])
          replaceind!(M[j],siteinds(M)[j][2],hB[j]')
        else
          replaceind!(M[j],siteinds(M)[j][1],siteinds(B.X)[j][1])
          replaceind!(M[j],siteinds(M)[j][2],siteinds(B.X)[j][2])
        end
      end
    end
  end
end

replacehilbertspace!(A::Union{MPS,MPO},B::LPDO) = 
  replacehilbertspace!(LPDO(A),B)

replacehilbertspace!(A::LPDO,B::Union{MPS,MPO}) = 
  replacehilbertspace!(A,LPDO(B))

replacehilbertspace!(A::Union{MPS,MPO},B::Union{MPS,MPO}) = 
  replacehilbertspace!(LPDO(A),LPDO(B))

