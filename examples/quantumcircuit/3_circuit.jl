using PastaQ

""" Build a circuit manually and run """
# Initialize the quantum state
N = 4
psi = qubits(N) 
# Define the gates data structure
gates = [
  (gate = "X" , site = 1),
  (gate = "Cx", site = [1,2]),
  (gate = "Rx", site = 2,params=(θ=0.5,)),
  (gate = "Rn", site = 3,params=(θ=0.5,ϕ=0.2,λ=1.2)),
  (gate = "Cz", site = [3,4])]

# Compile the gates into tensors
tensors = compilecircuit(psi,gates)

# Run the circuit (in-place)
runcircuit!(psi,tensors)

# Set the state to |0000...> (but keep the indices)
psi = resetqubits!(psi)

# Run the circuit without modifying initial state
psi_out = runcircuit(psi,tensors)

""" Build a circuit using layers and run """

gates = []

hadamardlayer!(gates,N)
rand1Qrotationlayer!(gates,N)
Cxlayer!(gates,N,sequence = "odd")
Cxlayer!(gates,N,sequence = "even")

tensors = compilecircuit(psi,gates)

runcircuit!(psi,tensors)

""" Use pre-made circuit """
depth = 4
gates = randomquantumcircuit(N,depth)
tensors = compilecircuit(psi,gates)
runcircuit!(psi,tensors)

""" Run circuit with different measurement bases """

psi = resetqubits!(psi)
nshots = 100
bases = generatemeasurementsettings(N,nshots)
#1000×4 Array{String,2}:
# "X"  "Y"  "Y"  "Y"
# "X"  "Y"  "Z"  "Y"
# "Z"  "Y"  "Y"  "Y"
# "Y"  "Z"  "Y"  "Z"
# ...
depth = 4
circuit_gates = randomquantumcircuit(N,depth)
circuit_tensors = compilecircuit(psi,circuit_gates)

samples = Matrix{Int64}(undef, nshots, N)

for n in 1:nshots
  meas_gates = makemeasurementgates(bases[1,:])
  meas_tensors = compilecircuit(psi,meas_gates)
  tensors = vcat(circuit_tensors,meas_tensors)
  psi_out = runcircuit(psi,tensors)
  #sample = measure(psi_out,1)
  samples[n,:] = measure(psi_out,1)
  println(bases[n,:],samples[n,:])
end
println("\n\n")

""" Run circuit with different preparation states / measurement bases """

psi = resetqubits!(psi)
nshots = 100
bases = generatemeasurementsettings(N,nshots)
#1000×4 Array{String,2}:
# "X"  "Y"  "Y"  "Y"
# "X"  "Y"  "Z"  "Y"
# "Z"  "Y"  "Y"  "Y"
# "Y"  "Z"  "Y"  "Z"
# ...
prep = generatepreparationsettings(N,nshots)
#1000×4 Array{String,2}:
# "Ym"  "Ym"  "Xm"  "Yp"
# "Xm"  "Zm"  "Yp"  "Zp"
# "Zm"  "Zm"  "Xm"  "Zm"
# "Xp"  "Xp"  "Yp"  "Yp"

depth = 4
circuit_gates = randomquantumcircuit(N,depth)
circuit_tensors = compilecircuit(psi,circuit_gates)

samples = Matrix{Int64}(undef, nshots, N)

for n in 1:nshots
  prep_gates = makepreparationgates(prep[n,:])
  prep_tensors = compilecircuit(psi,prep_gates)
  tensors = vcat(prep_tensors,circuit_tensors)

  meas_gates = makemeasurementgates(bases[n,:])
  meas_tensors = compilecircuit(psi,meas_gates)
  tensors = vcat(tensors,meas_tensors)
  
  psi_out = runcircuit(psi,tensors)
  samples[n,:] = measure(psi_out,1)
  println(prep[n,:],bases[n,:],samples[n,:])
end

