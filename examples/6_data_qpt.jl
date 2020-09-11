using ITensors
using PastaQ
using Random
using HDF5

Random.seed!(1234)

N =20
depth=4
nshots = 20000

gates = randomquantumcircuit(N,depth)
noise=nothing

if isnothing(noise)
  Φ = choimatrix(N,gates)
  
  (data_in,data_out) = generatedata(N,gates,nshots; process=true)
  
  output_path = "data_qpt_N$(N)_unitary.h5"
  h5open(output_path, "w") do file
    write(file,"data_in",data_in)
    write(file,"data_out",data_out)
    write(file,"choi",Φ)
  end    
else
  Λ = choimatrix(N,gates;noise="AD",γ=0.1)

  (data_in,data_out) = generatedata(N,gates,nshots;process=true,
                                   noise="AD",γ=0.1)

  output_path = "data_qpt_N$(N)_noisy.h5"
  h5open(output_path, "w") do file
    write(file,"data_in",data_in)
    write(file,"data_out",data_out)
    write(file,"choi",Λ)
  end    
end

