struct CircuitExperiment 
  N::Int
  seed::Int
  rng::MersenneTwister
  P::Vector{ITensor}
  M::Vector{ITensor}
  infos::Dict
end

# Constructor
function CircuitExperiment(;N::Int,seed::Int=1234)
  # Random number generator
  rng = MersenneTwister(seed)
  infos = Dict()
  P = ITensor[]
  M = ITensor[]
  return CircuitExperiment(N,seed,rng,P,M,infos)
end

function BuildPreparationBases!(experiment::CircuitExperiment,gates::QuantumGates;
                               prep_id::String="computational")
  if prep_id == "computational"
    experiment.infos["prep_id"] = "computational"
  elseif prep_id == "pauli6"
    # 1 -> prepare |0>
    # Gate: Identity
    push!(experiment.P,gates.Id)
    # 2 -> prepare |1>
    # Gate: X
    push!(experiment.P,gates.X)
    # 3 -> prepare |+>
    # Gate: H
    push!(experiment.P,gates.H)
    # 4 -> prepare |->
    # Gate: X H
    gate = gates.H * prime(gates.X)
    gate = setprime(gate,plev=2,1)
    push!(experiment.P,gate)
    # 5 -> prepare |l>
    # Gate: K
    push!(experiment.P,gates.Kp)
    # 6 -> prepare |->
    # Gate: X K
    gate = gates.Kp * prime(gates.X)
    gate = setprime(gate,plev=2,1)
    push!(experiment.P,gate)
    experiment.infos["prep_id"] = "pauli6"
  else
    error("Preparation set not recognized")
  end
end

function BuildMeasurementBases!(experiment::CircuitExperiment,gates::QuantumGates;
                               meas_id::String="computational")
  if meas_id == "computational"
    experiment.infos["meas_id"] = "computational"
  elseif meas_id == "pauli6"
    # 1 -> Z basis 
    # Gate: Identity
    push!(experiment.M,gates.Id)
    # 2 -> X basis
    # Gate: H
    push!(experiment.M,gates.H)
    # 3 -> Y basis
    # Gate: K
    push!(experiment.M,gates.Km)
    experiment.infos["meas_id"] = "pauli6"
  else
    error("Preparation set not recognized")
  end
end

# Prepare a product state of pauli eigenstates
function PrepareState(qc::QuantumCircuit,experiment::CircuitExperiment,state::Vector{Int})
  mps_state = InitializeQubits(qc)
  for j in 1:qc.N
    ApplySingleQubitGate!(mps_state,experiment.P[state[j]],j)
  end
  return mps_state
end

# Prepare a product state of pauli eigenstates
function RotateMeasurementBasis!(mps::MPS,qc::QuantumCircuit,experiment::CircuitExperiment,basis::Vector{Int})
  for j in 1:qc.N
    ApplySingleQubitGate!(mps,experiment.M[basis[j]],j)
  end
  return mps
end

function RunExperiment(qc::QuantumCircuit,gates::QuantumGates,experiment::CircuitExperiment;
                       prep_id::String="computational",
                       meas_id::String="computational",
                       nshots::Int)
  if prep_id == "computational" && meas_id == "computational"
    psi_in = InitializeQubits(qc)
    psi_out = ApplyCircuit(qc,psi_in)
    orthogonalize!(psi_out,1)
    measurements = Array[]
    for n in 1:nshots
      measurement = sample(psi_out)
      measurement .-= 1
      push!(measurements,measurement)
    end
    return measurements

  elseif prep_id == "pauli6" && meas_id == "computational"
    BuildPreparationBases!(experiment,gates,prep_id=prep_id)
    measurements = Array[]
    prep_states  = Array[]
    for n in 1:nshots
      state = rand(experiment.rng,1:6,qc.N)
      psi_in = PrepareState(qc,experiment,state)
      psi_out = ApplyCircuit(qc,psi_in)
      measurement = sample!(psi_out)
      measurement .-= 1
      push!(prep_states,state)
      push!(measurements,measurement)
    end
    return prep_states,measurements

  elseif prep_id == "computational" && meas_id == "pauli6"
    BuildMeasurementBases!(experiment,gates,meas_id=meas_id)
    measurements = Array[]
    meas_bases   = Array[]
    for n in 1:nshots
      basis = rand(experiment.rng,1:3,qc.N)
      psi_in = InitializeQubits(qc)
      psi = ApplyCircuit(qc,psi_in)
      psi_out = RotateMeasurementBasis!(psi,qc,experiment,basis)
      measurement = sample!(psi_out)
      measurement .-= 1
      push!(meas_bases,basis)
      push!(measurements,measurement)
    end  
    return meas_bases,measurements
  
  elseif prep_id == "pauli6" && meas_id == "pauli6"
    BuildPreparationBases!(experiment,gates,prep_id="pauli6")
    BuildMeasurementBases!(experiment,gates,meas_id="pauli6")
    measurements = Array[]
    prep_states  = Array[]
    meas_bases   = Array[]
    for n in 1:nshots
      state = rand(experiment.rng,1:6,qc.N)
      psi_in = PrepareState(qc,experiment,state)
      psi = ApplyCircuit(qc,psi_in)
      basis = rand(experiment.rng,1:3,qc.N)
      psi_out = RotateMeasurementBasis!(psi,qc,experiment,basis)
      measurement = sample!(psi_out)
      measurement .-= 1
      push!(prep_states,state)
      push!(meas_bases,basis)
      push!(measurements,measurement)
    end
    return prep_states,meas_bases,measurements
  end
  
end




