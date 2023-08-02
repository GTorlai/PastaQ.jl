"""

SAVE AND READ SAMPLES

"""

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
      write(fout, "state", "nothing")
    else
      write(fout, "state", model)
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
      write(fout, "state", "nothing")
    else
      write(fout, "state", model)
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
      write(fout, "state", "nothing")
    else
      write(fout, "state", model)
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
  if haskey(fin, "state")
    g = fin["state"]
    if haskey(attributes(g), "type")
      typestring = read(attributes(g)["type"])
      modeltype = eval(Meta.parse(typestring))
      model = read(fin, "state", modeltype)
    else
      model = read(fin, "state")
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
Various printing functionalities
"""

printmetric(name::String, metric::Int) = @printf("%s = %d  ", name, metric)
printmetric(name::String, metric::Float64) = @printf("%s = %-4.4f  ", name, metric)
printmetric(name::String, metric::AbstractArray) = @printf("%s = [...]  ", name)

function printmetric(name::String, metric::Complex)
  if imag(metric) < 1e-8
    @printf("%s = %-4.4f  ", name, real(metric))
  else
    @printf("%s = %.4f±i%-4.4f  ", name, real(metric), imag(metric))
  end
end

function printobserver(observer::DataFrame, print_metrics::Union{String,AbstractArray})
  if !isempty(print_metrics)
    if print_metrics isa String
      printmetric(print_metrics, results(observer, print_metrics)[end])
    else
      for metric in print_metrics
        printmetric(metric, results(observer, metric)[end])
      end
    end
  end
  return nothing
end
