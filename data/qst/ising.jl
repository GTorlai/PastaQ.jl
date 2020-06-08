using ITensors
using PastaQ
using Random
using HDF5

Random.seed!(1234)

N = 20

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

energy, psi = dmrg(H,psi0, sweeps)
#println("Final energy = $energy")

nshots = 50000

println("Generating data...")
bases = generatemeasurementsettings(N,nshots,bases_id=["X","Y","Z"])
samples = Matrix{Int64}(undef, nshots, N)

for n in 1:nshots
  meas_gates = makemeasurementgates(bases[n,:])
  meas_tensors = compilecircuit(psi,meas_gates)
  psi_out = runcircuit(psi,meas_tensors)
  samples[n,:] = measure(psi_out,1)
end

output_path = "data_ising_N$(N).h5"
h5open(output_path, "w") do file
  write(file,"samples",samples)
  write(file,"bases",bases)
  write(file,"psi",psi)
end

