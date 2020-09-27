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
    # The reference state describes a process, i.e. it 
    # contains output/intput indices.
    # ----------------------------------------
    # ----------------------------------------
    # TODO: update with unsplit version 
    if (ψ isa MPS) & (M isa MPS)
     # If both objects are MPS: 
     for j in 1:length(ψ)
        replaceind!(ψ[j],sψ[j],sM[j])
      end
    else
      error("not yet implemented")
    end
    # ----------------------------------------
    # ----------------------------------------
  else
    # The referencce state is either a MPS wavefunction
    # or a MPO density operator
    for j in 1:length(ψ)
      replaceind!(ψ[j],sψ[j],sM[j])
    end
  end
end

replacehilbertspace!(ψ::MPS,L::LPDO) = 
  replacehilbertspace!(LPDO(ψ),L)

replacehilbertspace!(ψ::MPS,M::Union{MPS,MPO}) = 
  replacehilbertspace!(LPDO(ψ),LPDO(M))

#replacehilbertspace!(Λ::LPDO{MPO},L::LPDO; split_noisyqpt::Bool=false) = 
#  replacehilbertspace(copy(Λ),L;split_noisyqpt=split_noisyqpt)

function replacehilbertspace!(Λ::LPDO{MPO},L::LPDO; split_noisyqpt::Bool=false)
  mpo = Λ.X  
  M = L.X
   
  # ----------------------------------------
  # ----------------------------------------
  # TODO: update with unsplit version (remove)
  if split_noisyqpt
    for j in 1:length(mpo)
      replaceind!(mpo[j],firstind(mpo[j],tags="Site",plev=0),firstind(M[j],tags="Site"))
      replaceind!(mpo[j],firstind(mpo[j],tags="Site",plev=1),firstind(M[j],tags="Site")')
    end
  else
    # ----------------------------------------
    # ----------------------------------------
    
    # Get site indices from the model
    h_M = hilbertspace(M)

    # Check if M describeds process (has input/output indices)
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
    # reference Hilbert space of a process. There should be a rule
    # whether the output/input indices are handled.
    if M_isaprocess & !mpo_isaprocess
      error("not yet implemented")
    end
    # 1. The reference state is a process (input/output indices)
    if M_isaprocess
      # a. The target state is a circuit MPO
      if !mpo_ispurified & mpo_isaprocess
        for j in 1:length(mpo)
          # Pick up input/output indices from the reference state
          replaceind!(mpo[j],firstind(mpo[j],tags="Output"),firstind(M[j],tags="Output"))
          replaceind!(mpo[j],firstind(mpo[j],tags="Input"),firstind(M[j],tags="Input"))
          # Set output (ket) plev=1 (since it is a regular MPO)
          setprime!(mpo[j],1,tags="Output")
        end
      # b. the target state is a choi matrix
      elseif mpo_ispurified & mpo_isaprocess
        for j in 1:length(mpo)
          # Get output and input indices from the reference proccess M
          replaceind!(mpo[j],firstind(mpo[j],tags="Output"),firstind(M[j],tags="Output"))
          replaceind!(mpo[j],firstind(mpo[j],tags="Input"),firstind(M[j],tags="Input"))
          # Unprime the output indicex (since it's a LPDO representation of the channel)
          noprime!(mpo[j],tags="Output")
        end        
      else
        error("Not yet implemented")
      end
    # 1. The reference state is not a process (has regular qubits tags)
    else
      # a. the reference state is a MPS/MPO 
      # E.g.: MPO circuit, MPS wavefunction,MPO density matrix
      if !mpo_ispurified & !mpo_isaprocess
        for j in 1:length(mpo)
          # Pick up target indices by the corresponding prime levels
          replaceind!(mpo[j],firstsiteinds(mpo,plev=0)[j],h_M[j])
          replaceind!(mpo[j],firstsiteinds(mpo,plev=1)[j],h_M[j]')
        end
      # b. the reference state is a purified MPO (e.g. LPDO)
      elseif mpo_ispurified & !mpo_isaprocess
        for j in 1:length(mpo)
          replaceind!(mpo[j],firstsiteinds(mpo,plev=0)[j],h_M[j])
        end
      else
        error("Not yet implemented")
      end
    end
  end
end

replacehilbertspace!(mpo::MPO,L::LPDO{MPO};split_noisyqpt=false) = 
  replacehilbertspace!(LPDO(mpo),L;split_noisyqpt=split_noisyqpt)

replacehilbertspace!(mpo::MPO,M::Union{MPS,MPO}) = 
  replacehilbertspace!(LPDO(mpo),LPDO(M))

replacehilbertspace!(Λ::LPDO{MPO},M::Union{MPS,MPO}) = 
  replacehilbertspace!(Λ,LPDO(M))

replacehilbertspace!(Λ::LPDO{MPS},M::Union{MPS,MPO}) = 
  replacehilbertspace!(Λ,LPDO(M))


#function replacehilbertspace!(Λ::LPDO{MPO},L::LPDO; split_noisyqpt::Bool=false)
#  mpo = Λ.X  
#  M = L.X
#  
#  # ----------------------------------------
#  # ----------------------------------------
#  # TODO: update with unsplit version (remove)
#  if split_noisyqpt
#    for j in 1:length(mpo)
#      replaceind!(mpo[j],firstind(mpo[j],tags="Site",plev=0),firstind(M[j],tags="Site"))
#      replaceind!(mpo[j],firstind(mpo[j],tags="Site",plev=1),firstind(M[j],tags="Site")')
#    end
#  # ----------------------------------------
#  # ----------------------------------------
#  else
#    # Get site indices from the model
#    h_M = hilbertspace(M)
#    # Check if M describeds process (has input/output indices)
#    M_isaprocess   = any(x -> hastags(x,"Input") , M)
#    # Check if mpo represents a quantum channel
#    mpo_ispurified = any(x -> hastags(x,"Purifier") , mpo)
#    # Check if mpo is mixed (either state (density-matrix) or process (choi-matrix))
#    mpo_isaprocess = any(x -> hastags(x,"Input") , mpo)
#    # Check that process tags (input/output) are set properly
#    if M_isaprocess
#      @assert any(x -> hastags(x,"Output") , M)
#    end
#    if mpo_isaprocess
#      @assert any(x -> hastags(x,"Output") , mpo)
#    end
#    # Hilbertspace replacement not imlemented for a state given a
#    # reference Hilbert space of a process. There should be a rule
#    # whether the output/input indices are handled.
#    if M_isaprocess & !mpo_isaprocess
#      error("not yet implemented")
#    end
#    # 1. the target state `mpo` is NOT purified. It has two sets
#    # of Site indices (primed and unprimed for bra and ket respectively)
#    if !mpo_ispurified
#      # a. Both reference and target describe a process (input/output tags)
#      if M_isaprocess & mpo_isaprocess
#        for j in 1:length(mpo)
#          # Pick up input/output indices from the reference state
#          replaceind!(mpo[j],firstind(mpo[j],tags="Output"),firstind(M[j],tags="Output"))
#          replaceind!(mpo[j],firstind(mpo[j],tags="Input"),firstind(M[j],tags="Input"))
#          # Set output plev=1 (in case the reference state is a purified-MPO, 
#          # where both input and output indices are unprimed by default.
#          setprime!(mpo[j],1,tags="Output")
#        end
#      # b. Both reference and target describe a state (regular `qubit` tags).
#      elseif !M_isaprocess & !mpo_isaprocess
#        for j in 1:length(mpo)
#          # Pick up target indices by the corresponding prime levels
#          replaceind!(mpo[j],firstsiteinds(mpo,plev=0)[j],h_M[j])
#          replaceind!(mpo[j],firstsiteinds(mpo,plev=1)[j],h_M[j]')
#        end
#      else
#        error("Not yet implemented")
#      end
#    # mpo has a purified index (is a LPDO)
#    # 2. The target state is a purified-MPO (has one set of site indices
#    # and one set of purification indices (both unprimed).
#    else
#      # Purified MPO
#      for j in 1:length(mpo)
#        if M_isaprocess
#          replaceind!(mpo[j],firstind(mpo[j],tags="Output"),firstind(M[j],tags="Output")')
#          replaceind!(mpo[j],firstind(mpo[j],tags="Input"),firstind(M[j],tags="Input"))
#          if mpo_ispurified
#            noprime!(mpo[j])
#          end
#        elseif !mpo_isaprocess
#          replaceind!(mpo[j],firstsiteinds(mpo,plev=0)[j],h_M[j])
#        else
#          error("Not yet implemented")
#        end
#      end
#    end
#  end
#  return Λ
#end


