using ITensors
using PastaQ
using Random
using HDF5

Random.seed!(1234)
N = 20
depth=2
noise=nothing
nshots = nothing

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
  
  #opt = SGD(η = 0.01)
  opt = Adagrad(Ψ0;η=0.005,ϵ=1E-8)

  Ψ = processtomography(Ψ0,data_in,data_out,opt;
                        χ = χ,
                        mixed=false,
                        batchsize=900,
                        epochs=1000,
                        target=Φ,
                        localnorm=true,
                        globalnorm=false)
  
else
  input_path = "data_qpt_N$(N)_noisy.h5"
  data_in,data_out,Λ = loadtrainingdataQPT(input_path; ismpo=true)
  χ = maxlinkdim(Λ)
  if !isnothing(nshots)
    data_in  = data_in[1:nshots,:]    
    data_out = data_out[1:nshots,:]
  end
  @show maxlinkdim(Φ)
  
  ##@show Λ
  #for j in 1:length(Λ) 
  #  ind = (j+1)÷2
  #  if isodd(j)
  #    replacetags!(Λ[j],"Input,Site,n=$ind,qubit","Site,n=$j),qubit")
  #  else
  #    replacetags!(Λ[j],"Output,Site,n=$ind,qubit","Site,n=$j,qubit")
  #  end
  #end
  #opt = SGD(η = 0.1)
  #
  #Ψ = processtomography(data_in,data_out,opt;
  #                      χ = χ,
  #                      mixed=true,
  #                      batchsize=100,
  #                      epochs=100,
  #                      target=Λ,
  #                      localnorm=false,
  #                      globalnorm=false)
  #
end
