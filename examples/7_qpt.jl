using ITensors
using PastaQ
using Random
using HDF5

Random.seed!(1234)
N = 4
depth=2
noise= true#nothing
nshots = 10000

if isnothing(noise)
  input_path = "data_qpt_N$(N)_unitary_random_D=$(depth).h5"
  data_in,data_out,Φ = loadtrainingdataQPT(input_path)
  if !isnothing(nshots)
    data_in  = data_in[1:nshots,:]    
    data_out = data_out[1:nshots,:]
  end
  χ = 2*maxlinkdim(Φ)
  @show maxlinkdim(Φ)

  Ψ0 = initializetomography(2*N,χ,σ=0.1)
  
  opt = Sgd(Ψ0; η = 0.1,γ=0.9)
  
  obs = TomographyObserver(Ψ0)
  outputpath= "qpt_N$(N)_unitary_random_D=$(depth)_SGD_ns$(nshots).h5"
  (Ψ,obs) = processtomography(Ψ0,data_in,data_out,opt;
                        observer=obs,
                        fout=outputpath,
                        batchsize=1000,
                        epochs=1000,
                        target=Φ,
                        localnorm=true,
                        globalnorm=false)
  
else
  input_path = "data_qpt_N$(N)_noisy_random_D=$(depth).h5"
  data_in,data_out,Φ = loadtrainingdataQPT(input_path; ismpo=true)
  if !isnothing(nshots)
    data_in  = data_in[1:nshots,:]    
    data_out = data_out[1:nshots,:]
  end
  @show maxlinkdim(Φ)
  
  opt = Sgd(Λ0; η = 0.1,γ=0.9)
  χ = maxlinkdim(Φ)
  ξ = 2  
  Λ0 = initializetomography(2*N,χ,ξ)
  outputpath=nothing
  (obs,Λ)  = processtomography(Λ0,data_in,data_out,opt;
                               observer=obs,
                               fout=outputpath,
                               batchsize=1000,
                               epochs=1000,
                               target=Φ,
                               localnorm=true,
                               globalnorm=false)
end
