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
  if haskey(fin, "inputs")
    inputs = read(fin, "inputs")
    bases = read(fin, "bases")
    outcomes = read(fin, "outcomes")
    data = inputs .=> (bases .=> outcomes)
    # Measurements in bases
  elseif haskey(fin, "bases")
    bases = read(fin, "bases")
    outcomes = read(fin, "outcomes")
    data = bases .=> outcomes
    # Measurements in Z basis
  elseif haskey(fin, "outcomes")
    data = read(fin, "outcomes")
  else
    close(fin)
    error(
      "File must contain either \"data\" for quantum state tomography data or \"data_first\" and \"data_second\" for quantum process tomography.",
    )
  end

  # Check if a model is saved, if so read it and return it
  if haskey(fin, "model")
    g = fin["model"]

    if haskey(attributes(g), "type")
      typestring = read(attributes(g)["type"])
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
function writesamples(
  data::Matrix{Int}, model::Union{MPS,MPO,LPDO,Nothing}, output_path::String
)
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

function writesamples(data::Matrix{Int}, output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "outcomes", data)
  end
end

function writesamples(
  data::Matrix{Pair{String,Int}}, model::Union{MPS,MPO,LPDO,Nothing}, output_path::String
)
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

function writesamples(data::Matrix{Pair{String,Int}}, output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "bases", first.(data))
    write(fout, "outcomes", last.(data))
  end
end

function writesamples(
  data::Matrix{Pair{String,Pair{String,Int}}},
  model::Union{MPS,MPO,LPDO,Nothing},
  output_path::String,
)
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

function writesamples(data::Matrix{Pair{String,Pair{String,Int}}}, output_path::String)
  # Make the path the file will sit in, if it doesn't exist
  mkpath(dirname(output_path))
  h5rewrite(output_path) do fout
    write(fout, "inputs", first.(data))
    write(fout, "bases", first.(last.(data)))
    write(fout, "outcomes", last.(last.(data)))
  end
end

function savetomographyobserver(observer::Observer, output_path::String; model = nothing)
  mkpath(dirname(output_path)) 
  
  h5rewrite(output_path) do fout
    if !isnothing(model)
      write(fout, "model", model)
    else
      write(fout, "model", "nothing")
    end
    
    params = results(observer, "parameters")
    g1 = create_group(fout, "parameters")
    g1["batchsize"] = params["batchsize"] 
    g1["nshots"] = params["dataset_size"] 
    g1["measurement_frequency"] = params["measurement_frequency"]
    g1["optimizer"] = params["optimizer"][:name]
    g1["learning_rate"] = params["optimizer"][:η]
    attributes(g1)["Description"] = "This group contains the training parameters."
    
    g2 = create_group(fout, "measurements")
    for (measurement, value) in observer.measurements
      if measurement != "parameters"
        g2[measurement] = real.(last(value))
      end
    end
    attributes(g2)["Description"] = "This group contains measurements." 
  end
end

#function readtomographyobserver(input_path::String)
#  fin = h5open(input_path, "r")
#  
#  parameters   = read(fin["parameters"])
#  measurements = read(fin["measurements"])
#
#  # Check if a model is saved, if so read it and return it
#  if haskey(fin, "model")
#    g = fin["model"]
#
#    if haskey(attributes(g), "type")
#      typestring = read(attributes(g)["type"])
#      modeltype = eval(Meta.parse(typestring))
#      model = read(fin, "model", modeltype)
#    else
#      model = read(fin, "model")
#      if model == "nothing"
#        model = nothing
#      else
#        error("model must be MPS, LPDO, or Nothing")
#      end
#    end
#    close(fin)
#    #return parameters, measurements, model
#  end
#  close(fin)
#  return parameters, measurements
#end
