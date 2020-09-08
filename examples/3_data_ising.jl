using ITensors
using PastaQ
using Random
using HDF5

Random.seed!(1234)
N = 100
h = 1.0
target = "GS"

if target == "GS"
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

elseif target == "thermal"
  β = 1.0
  τ = 0.01
  
  ρ = transversefieldising(N,h,β=β,τ=τ) 
  
  println("Generating data...")
  bases = generatemeasurementsettings(N,nshots,bases_id=["X","Y","Z"])
  data = generatedata(ρ,nshots,bases)
  
  output_path = "data_ising_N$(N)_beta=1.0.h5"
  
  h5open(output_path, "w") do file
    write(file,"data",data)
    write(file,"rho",ρ)
  end
end
