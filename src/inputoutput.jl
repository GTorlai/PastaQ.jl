"""
    readsamples(input_path::String)

Load data and model from file:

# Arguments:
  - `input_path`: path to file
"""
function readsamples(input_path::String)
  fin = h5open(input_path, "r")
  # Check if the data is for state tomography or process tomography
  # Process tomography
  if exists(fin, "inputs")
    inputs = read(fin, "inputs")
    bases = read(fin, "bases")
    outcomes = read(fin,"outcomes")
    data = inputs .=> (bases .=> outcomes)
  # Measurements in bases
  elseif exists(fin, "bases") 
    bases = read(fin, "bases")
    outcomes = read(fin,"outcomes")
    data = bases .=> outcomes
  # Measurements in Z basis
  elseif exists(fin, "outcomes")
    data = read(fin, "outcomes")
  else
    close(fin)
    error("File must contain either \"data\" for quantum state tomography data or \"data_first\" and \"data_second\" for quantum process tomography.")
  end

  # Check if a model is saved, if so read it and return it
  if exists(fin, "model")
    g = fin["model"]

    if exists(attrs(g), "type")
      typestring = read(attrs(g)["type"])
      modeltype = eval(Meta.parse(typestring))
      model = read(fin, "model", modeltype)
    else
      model = read(fin, "model")
      if model == "nothing"
        model = nothing
      else
        error("model must be MPS, LPDO, or Nothing")
      end
    end
    close(fin)
    return data, model
  end

  close(fin)
  return data
end


"""
    writesamples(data::Matrix{Int},
                 [model::Union{MPS, MPO, LPDO, Nothing},]
                 output_path::String)

Save data and model on file:

# Arguments:
  - `data`: array of measurement data
  - `model`: (optional) MPS, MPO, or Choi
  - `output_path`: path to file
"""
function writesamples(data::Matrix{Int},
                      model::Union{MPS, MPO, LPDO, Nothing},
                      output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "outcomes", data)
    if isnothing(model)
      write(fout, "model", "nothing")
    else
      write(fout, "model", model)
    end
  end
end

function writesamples(data::Matrix{Int},
                      output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "outcomes", data)
  end
end

function writesamples(data::Matrix{Pair{String, Int}},
                      model::Union{MPS, MPO, LPDO, Nothing},
                      output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "bases", first.(data))
    write(fout, "outcomes", last.(data))
    if isnothing(model)
      write(fout, "model", "nothing")
    else
      write(fout, "model", model)
    end
  end
end

function writesamples(data::Matrix{Pair{String, Int}},
                      output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "bases", first.(data))
    write(fout, "outcomes", last.(data))
  end
end

function writesamples(data::Matrix{Pair{String, Pair{String, Int}}},
                      model::Union{MPS, MPO, LPDO, Nothing},
                      output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "inputs", first.(data))
    write(fout, "bases", first.(last.(data)))
    write(fout, "outcomes", last.(last.(data)))
    if isnothing(model)
      write(fout, "model", "nothing")
    else
      write(fout, "model", model)
    end
  end
end

function writesamples(data::Matrix{Pair{String, Pair{String, Int}}},
                      output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "inputs", first.(data))
    write(fout, "bases", first.(last.(data)))
    write(fout, "outcomes", last.(last.(data)))
  end
end




