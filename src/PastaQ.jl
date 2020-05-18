module PastaQ 

using ITensors
using Random
using LinearAlgebra
using JLD
using HDF5

export 
  QuantumGates,
  U3,
  cX
export 
  QuantumCircuit,
  PopulateInfoDict!,
  ApplySingleQubitGate!,
  ApplyTwoQubitGate!,
  InitializeQubits,
  RandomSingleQubitLayer!,
  LoadQuantumCircuit,
  ApplyCircuit

export
  CircuitExperiment,
  BuildPreparationBases!,
  BuildMeasurementBases!,
  PrepareState,
  RotateMeasurementBasis!,
  RunExperiment

export
  FullMatrix

include("utils.jl")
include("quantumcircuit/quantumgates.jl")
include("quantumcircuit/quantumcircuit.jl")
include("quantumcircuit/circuitexperiment.jl")
end
