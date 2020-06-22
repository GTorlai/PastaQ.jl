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
  ampo .+=(-2.0,"Sz",j,"Sz",j+1)
  ampo .+=(-1.0,"Sx",j) 
end
ampo += (-1.0,"Sx",N)
H = MPO(ampo,sites)

psi0 = randomMPS(sites)

sweeps = Sweeps(50)
maxdim!(sweeps, 10,20,100,100,200)
cutoff!(sweeps, 1E-10)

energy, psi = dmrg(H,psi0, sweeps)
#println("Final energy = $energy")

nshots = 50000

println("Generating data...")
bases = generatemeasurementsettings(N,nshots,bases_id=["X","Y","Z"])

data = generatedata(psi,nshots,bases)

output_path = "data_ising_N$(N).h5"

h5open(output_path, "w") do file
  write(file,"data",data)
  write(file,"psi",psi)
end

