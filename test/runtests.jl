using Test

# TODO: replace with:
#
# filenames = readdir()
#
# However this seems to cause some issues with some process
# tomography tests, it seems to be sensitive to the ordering
# they are called.
# filenames = [
#   "test_array.jl",
#   "test_autodiff.jl",
#   "test_circuits.jl",
#   "test_distances.jl",
#   "test_fulltomography.jl",
#   "test_gates.jl",
#   "test_getsamples.jl",
#   "test_io.jl",
#   "test_noise.jl",
#   "test_optimizers.jl",
#   "test_processtomography.jl",
#   "test_productstates.jl",
#   "test_qubitarrays.jl",
#   "test_randomstates.jl",
#   "test_runcircuit.jl",
#   "test_statetomography.jl",
#   "test_utils.jl",
#   "test_gpu.jl",
# ]

# Implement a single function to read all files with a filter  
# all tests now pass, at least with a single thread.
# Also, removed the condition to check if filename starts with "test_" as it is guaranteed by the filter used in searchDir().
# Can upgrade filter to broadcast

searchDir(path, key) = filter(x->(all([ occursin(k,x) for k in key ]) ), readdir(path))
filenames = searchDir("./",["test_",".jl"])

@testset "PastaQ.jl" begin
  @testset "$filename" for filename in filenames
    # if endswith(filename, ".jl") #&& startswith(filename, "test_") 
      println("Running $filename")
      @time include(filename)
      # end 
  end

    #@time include("test_io.jl")

end
