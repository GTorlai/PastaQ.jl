module ITensorsGateEvolution

using ITensors

export ProductOps,
       ops,
       apply,
       findsiteinds,
       firstsiteinds,
       movesite,
       movesites,
       # TODO: remove this once ITensors v0.1.20 is registered
       filterinds

# TODO: remove this once ITensors v0.1.20 is registered
const filterinds = inds

# Extensions to ITensors
include("itensors/indexset.jl")
include("itensors/mps.jl")

# ProductOps type
include("productops.jl")

# Qubit site type
include("qubit.jl")

# Apply function
include("apply.jl")

end
