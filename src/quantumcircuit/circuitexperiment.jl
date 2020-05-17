struct CircuitExperiment 
  N::Int
  seed::Int
  rng::MersenneTwister
  P::Vector{ITensor}
  M::Vector{ITensor}
  nshots::Int
  infos::Dict
end

# Constructor
function CircuitExperiment(;N::Int,seed::Int=1234,nshots::Int=100)
  # Random number generator
  rng = MersenneTwister(seed)
  nshots = nshots
  infos = Dict()
  P = ITensor[]
  M = ITensor[]
  return CircuitExperiment(N,seed,rng,P,M,nshots,infos)
end

function BuildStatePreparation!(experiment::CircuitExperiment,gates::QuantumGates;
                               prep_id::String="computational")
  if prep_id == "computational"
    nothing
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
    push!(experiment.P,gates.K)
    # 6 -> prepare |->
    # Gate: X K
    gate = gates.K * prime(gates.X)
    gate = setprime(gate,plev=2,1)
    push!(experiment.P,gate)
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

