using ITensors
using PastaQ
using Random
using HDF5

Random.seed!(1234)

N = 4
depth = 2
nshots = 10000

gates = randomquantumcircuit(N,depth)
noise = nothing

if isnothing(noise)
  
  (Φ,data_in,data_out) = generate_processdata(N,gates,nshots;choi=true,return_state=true)
  
  output_path = "data_qpt_N$(N)_unitary_random_D=$(depth).h5"
  h5open(output_path, "w") do file
    write(file,"data_in",data_in)
    write(file,"data_out",data_out)
    write(file,"choi",Φ)
  end    
else
  
  (Λ,data_in,data_out) = generate_processdata(N,gates,nshots;
                                            noise="AD",γ=0.1)

  output_path = "data_qpt_N$(N)_noisy.h5"
  h5open(output_path, "w") do file
    write(file,"data_in",data_in)
    write(file,"data_out",data_out)
    write(file,"choi",Λ)
  end    
end

