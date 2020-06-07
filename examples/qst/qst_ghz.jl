using PastaQ
using Random

Random.seed!(123456)

N = 3

ghz = qubits(N)

applygate!(ghz,"H",1)
applygate!(ghz,"Cx",[1,2])
applygate!(ghz,"Cx",[2,3])

nshots = 1000
bases = generatemeasurementsettings(N,nshots)
samples = Matrix{Int64}(undef, nshots, N)

for n in 1:nshots
  meas_gates = makemeasurementgates(bases[n,:])
  meas_tensors = compilecircuit(ghz,meas_gates)
  psi_out = runcircuit(ghz,meas_tensors)
  samples[n,:] = measure(psi_out,1)
end

χ = 2
psi = initializeQST(N,χ)
opt = Optimizer(η = 0.01)
statetomography!(psi,opt,
                samples = samples,
                bases = bases,
                batchsize=500,
                epochs=1000,
                target=ghz,
                localnorm=true)

#χ = 2
#ξ = 2
#lpdo = initializeQST(N,χ,ξ)
#opt = Optimizer(η = 0.01)
#statetomography!(lpdo,opt,
#                samples = samples,
#                bases = bases,
#                batchsize=500,
#                epochs=1000,
#                target=ghz,
#                localnorm=true)
#
