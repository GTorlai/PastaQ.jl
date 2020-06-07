using ITensors
using PastaQ
using Random
using HDF5

Random.seed!(1234)

N = 30

sites = [Index(2; tags="Site, n=$s") for s in 1:N]
links = [Index(2; tags="Link, l=$l") for l in 1:N-1]

corner = zeros(2,2)
bulk   = zeros(2,2,2)

corner[1,1] = 2^(-1.0/(2*N))
corner[2,2] = 2^(-1.0/(2*N))
bulk[1,1,1] = 2^(-1.0/(2*N))
bulk[2,2,2] = 2^(-1.0/(2*N))

M = ITensor[]

push!(M,ITensor(corner,sites[1],links[1]))
for j in 2:N-1
  push!(M,ITensor(bulk,links[j-1],sites[j],links[j]))
end
push!(M,ITensor(corner,sites[N],links[N-1]))

psi = MPS(M)

nshots = 50000
bases = generatemeasurementsettings(N,nshots,bases_id=["X","Y","Z"])
samples = Matrix{Int64}(undef, nshots, N)

for n in 1:nshots
  meas_gates = makemeasurementgates(bases[n,:])
  meas_tensors = compilecircuit(psi,meas_gates)
  psi_out = runcircuit(psi,meas_tensors)
  samples[n,:] = measure(psi_out,1)
end

output_path = "data_ghz_N$(N).h5"
h5open(output_path, "w") do file
  write(file,"samples",samples)
  write(file,"bases",bases)
  write(file,"psi",psi)
end

