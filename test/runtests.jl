using Test

@testset "PastaQ.jl" begin
  @testset "$filename" for filename in readdir()
    if startswith(filename, "test_") && endswith(filename, ".jl")
      println("Running $filename")
      @time include(filename)
    end
  end
end
