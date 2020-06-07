using ITensors
using PastaQ
using Random
Random.seed!(123456)

N = 50
sites = siteinds("S=1/2",N)

# Get the Ising ground state
ampo = AutoMPO()
for j=1:N-1
  ampo .+=(-1.0,"Sz",j,"Sz",j+1)
  ampo .+=(-1.0,"Sx",j) 
end
ampo += (-1.0,"Sx",N)
H = MPO(ampo,sites)

psi0 = randomMPS(sites)

sweeps = Sweeps(5)
maxdim!(sweeps, 10,20,100,100,200)
cutoff!(sweeps, 1E-10)

energy, psi_ising = dmrg(H,psi0, sweeps)
println("Final energy = $energy")

nshots = 50000
println("Generating data...")
bases = generatemeasurementsettings(N,nshots)
samples = Matrix{Int64}(undef, nshots, N)
for n in 1:nshots
  meas_gates = makemeasurementgates(bases[n,:])
  meas_tensors = compilecircuit(psi_ising,meas_gates)
  psi_out = runcircuit(psi_ising,meas_tensors)
  samples[n,:] = measure(psi_out,1)
end

χ = maxlinkdim(psi_ising)
psi = initializeQST(N,χ,σ=0.1)
opt = Sgd(η = 0.05)
println("Training...")
statetomography!(psi,opt,
                 samples = samples,
                 bases = bases,
                 batchsize=1000,
                 epochs=1000,
                 target=psi_ising,
                 localnorm=true)
