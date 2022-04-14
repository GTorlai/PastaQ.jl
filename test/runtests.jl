using Test

# TODO: replace with:
#
# filenames = readdir()
#
# However this seems to cause some issues with some process
# tomography tests, it seems to be sensitive to the ordering
# they are called.
filenames = [
  "test_array.jl",
  "test_autodiff.jl",
  "test_circuits.jl",
  "test_distances.jl",
  "test_fulltomography.jl",
  "test_gates.jl",
  "test_getsamples.jl",
  "test_io.jl",
  "test_noise.jl",
  "test_optimizers.jl",
  "test_processtomography.jl",
  "test_productstates.jl",
  "test_qubitarrays.jl",
  "test_randomstates.jl",
  "test_runcircuit.jl",
#  "test_gpu.jl",
  "test_statetomography.jl",
  "test_utils.jl",
]

@testset "PastaQ.jl" begin
  @testset "$filename" for filename in filenames
    if startswith(filename, "test_") && endswith(filename, ".jl")
      println("Running $filename")
      @time include(filename)
    end
  end
end
