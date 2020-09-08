using ITensors
using PastaQ
using Random
using HDF5

Random.seed!(1234)

N = 100
h = 1.0

ψ = transversefieldising(N,h,iter=20)

nshots = 50000
println("Generating data...")
bases = generatemeasurementsettings(N,nshots,bases_id=["X","Y","Z"])

data = generatedata(ψ,nshots,bases)

output_path = "data_ising_N$(N).h5"

h5open(output_path, "w") do file
  write(file,"data",data)
  write(file,"psi",ψ)
end
