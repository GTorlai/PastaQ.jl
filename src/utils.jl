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


# TEMPORARY FUNCTION
# TODO: remove when `firstsiteinds(ψ::MPS)` is implemented
function hilbertspace(L::LPDO) 
  return  (L.X isa MPS ? siteinds(L.X) : firstsiteinds(L.X))
end

hilbertspace(M::Union{MPS,MPO}) = hilbertspace(LPDO(M))


function replacehilbertspace!(Λ::LPDO{MPS},L::LPDO)
  ψ = Λ.X
  M = L.X
  M_isaprocess = any(x -> hastags(x,"Input") , L.X)
  sM = hilbertspace(L)
  sψ = hilbertspace(ψ)
  if M_isaprocess 
    if (ψ isa MPS) & (M isa MPS)
     for j in 1:length(ψ)
        replaceind!(ψ[j],sψ[j],sM[j])
      end
    else
      error("not yet implemented")
    end
  else
    for j in 1:length(ψ)
      replaceind!(ψ[j],sψ[j],sM[j])
    end
  end
end

function replacehilbertspace!(Λ::LPDO{MPO},L::LPDO; split_noisyqpt::Bool=false)
  mpo = Λ.X  
  M = L.X
  h_mpo = hilbertspace(mpo)
  h_M = hilbertspace(M)

  # Check if M represents a quantum channel
  M_isaprocess   = any(x -> hastags(x,"Input") , M)
  # Check if mpo represents a quantum channel
  mpo_ispurified = any(x -> hastags(x,"Purifier") , mpo)
  # Check if mpo is mixed (either state (density-matrix) or process (choi-matrix))
  mpo_isaprocess = any(x -> hastags(x,"Input") , mpo)

  # Check that process tags (input/output) are set properly
  if M_isaprocess
    @assert any(x -> hastags(x,"Output") , M)
  end
  if mpo_isaprocess
    @assert any(x -> hastags(x,"Output") , mpo)
  end
  # Hilbertspace replacement not imlemented for a state given a
  # reference Hilbert state of a process
  if M_isaprocess & !mpo_isaprocess
    error("not yet implemented")
  end
  # mpo is a regular MPO (no purification index)
  #@show M_isaprocess, mpo_isaprocess,mpo_ispurified
  if !mpo_ispurified
    for j in 1:length(mpo)
      # Both reference and target object represents quantum channels
      if split_noisyqpt
        replaceind!(mpo[j],firstind(mpo[j],tags="Site",plev=0),firstind(M[j],tags="Site"))
        replaceind!(mpo[j],firstind(mpo[j],tags="Site",plev=1),firstind(M[j],tags="Site")')
      elseif M_isaprocess & mpo_isaprocess
        replaceind!(mpo[j],firstind(mpo[j],tags="Output"),firstind(M[j],tags="Output"))
        replaceind!(mpo[j],firstind(mpo[j],tags="Input"),firstind(M[j],tags="Input"))
        setprime!(mpo[j],1,tags="Output")
      elseif !M_isaprocess & !mpo_isaprocess
        replaceind!(mpo[j],firstsiteinds(mpo,plev=0)[j],h_M[j])
        replaceind!(mpo[j],firstsiteinds(mpo,plev=1)[j],h_M[j]')
      end
    end
  # mpo has a purified index (is a LPDO)
  else
    # Purified MPO
    for j in 1:length(mpo)
      if M_isaprocess
        replaceind!(mpo[j],firstind(mpo[j],tags="Output"),firstind(M[j],tags="Output")')
        replaceind!(mpo[j],firstind(mpo[j],tags="Input"),firstind(M[j],tags="Input"))
        if mpo_ispurified
          noprime!(mpo[j])
        end
      else
        replaceind!(mpo[j],firstsiteinds(mpo,plev=0)[j],h_M[j])
      end
    end
  end
end


replacehilbertspace!(ψ::MPS,L::LPDO) = 
  replacehilbertspace!(LPDO(ψ),L)

replacehilbertspace!(ψ::MPS,M::Union{MPS,MPO}) = 
  replacehilbertspace!(LPDO(ψ),LPDO(M))

replacehilbertspace!(mpo::MPO,L::LPDO;split_noisyqpt=false) = 
  replacehilbertspace!(LPDO(mpo),L;split_noisyqpt=split_noisyqpt)

replacehilbertspace!(mpo::MPO,M::Union{MPS,MPO}) = 
  replacehilbertspace!(LPDO(mpo),LPDO(M))

replacehilbertspace!(Λ::LPDO{MPO},M::Union{MPS,MPO}) = 
  replacehilbertspace!(Λ,LPDO(M))

replacehilbertspace!(Λ::LPDO{MPS},M::Union{MPS,MPO}) = 
  replacehilbertspace!(Λ,LPDO(M))

replacehilbertspace!(mpo::MPO,L::LPDO) = 
  replacehilbertspace!(LPDO(mpo),L)

#function replacehilbertspace!(mpo::MPO,L::LPDO)
#  mpo = Λ.X  
#  M = L.X
#  h_mpo = hilbertspace(mpo)
#  h_M = hilbertspace(M)
#
#  # Check if M represents a quantum channel
#  M_isaprocess   = any(x -> hastags(x,"Input") , M)
#  # Check if mpo represents a quantum channel
#  mpo_ispurified = any(x -> hastags(x,"Purifier") , mpo)
#  # Check if mpo is mixed (either state (density-matrix) or process (choi-matrix))
#  mpo_isaprocess = any(x -> hastags(x,"Input") , mpo)
#
#  # Check that process tags (input/output) are set properly
#  if M_isaprocess
#    @assert any(x -> hastags(x,"Output") , M)
#  end
#  if mpo_isaprocess
#    @assert any(x -> hastags(x,"Output") , mpo)
#  end
#  @show M
#  @show mpo 
#  # Hilbertspace replacement not imlemented for a state given a
#  # reference Hilbert state of a process
#  if M_isaprocess & !mpo_isaprocess
#    error("not yet implemented")
#  end
#  # mpo is a regular MPO (no purification index)
#  @show M_isaprocess, mpo_isaprocess,mpo_ispurified
#  if !mpo_ispurified
#    for j in 1:length(mpo)
#      # Both reference and target object represents quantum channels
#      if M_isaprocess & mpo_isaprocess
#        replaceind!(mpo[j],firstind(mpo[j],tags="Output"),firstind(M[j],tags="Output"))
#        replaceind!(mpo[j],firstind(mpo[j],tags="Input"),firstind(M[j],tags="Input"))
#        setprime!(mpo[j],1,tags="Output")
#      elseif !M_isaprocess & !mpo_isaprocess
#        replaceind!(mpo[j],firstsiteinds(mpo,plev=0)[j],h_M[j])
#        replaceind!(mpo[j],firstsiteinds(mpo,plev=1)[j],h_M[j]')
#      end
#    end
#  # mpo has a purified index (is a LPDO)
#  else
#    # Purified MPO
#    for j in 1:length(mpo)
#      if M_isaprocess
#        replaceind!(mpo[j],firstind(mpo[j],tags="Output"),firstind(M[j],tags="Output")')
#        replaceind!(mpo[j],firstind(mpo[j],tags="Input"),firstind(M[j],tags="Input"))
#        if mpo_ispurified
#          noprime!(mpo[j])
#        end
#      else
#        replaceind!(mpo[j],firstsiteinds(mpo,plev=0)[j],h_M[j])
#      end
#    end
#  end
#end
#
