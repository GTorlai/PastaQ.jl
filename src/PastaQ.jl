module PastaQ

using DataFrames: DataFrame
using HDF5
using ITensors
using JLD2
using LinearAlgebra
using Observers
using Optimisers: Optimisers
using Printf
using Random
using StatsBase: StatsBase, Weights

include("imports.jl")
include("exports.jl")
include("lpdo.jl")
include("itensor.jl")
include("distances.jl")
include("circuits/gates.jl")
include("circuits/noise.jl")
include("circuits/qubitarrays.jl")
include("circuits/circuits.jl")
include("circuits/runcircuit.jl")
include("circuits/getsamples.jl")
include("circuits/trottersuzuki.jl")
include("optimizers.jl")
include("productstates.jl")
include("randomstates.jl")
include("measurements.jl")
# included the file and added SCS and Convex to the manifest. Module importing `SCS.MathOptInterface` also does not work, and it needs to be installed and imported separately. This has also been implemented.
# fixed relevant tests as well.
include("tomography/fulltomography.jl")
include("tomography/tensornetwork-statetomography.jl")
include("tomography/tensornetwork-processtomography.jl")
include("tomography/quantumtomography.jl")
include("io.jl")
include("array.jl")
include("utils.jl")

# using Requires
# function __init__()
#   @require SCS = "c946c3f1-0d1f-5ce8-9dea-7daa1f7e2d13" begin
#     @require Convex = "f65535da-76fb-5f13-bab9-19810c17039a" include(
#       "tomography/fulltomography.jl"
#     )
#   end
# end

end # module
