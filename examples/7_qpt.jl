using ITensors
using PastaQ
using Random
using HDF5

Random.seed!(1234)
N = 4
depth=2
noise=nothing

if isnothing(noise)
  input_path = "data_qpt_N$(N)_unitary_random_D=$(depth).h5"
  data_in,data_out,Φ = loadtrainingdataQPT(input_path)
  χ = maxlinkdim(Φ)
  
  Ψ0 = initializetomography(2*N,χ,σ=0.1)
  
  #opt = SGD(η = 0.01)
  opt = Adagrad(Ψ0;η=0.01,ϵ=1E-8)

  Ψ = processtomography(Ψ0,data_in,data_out,opt;
                        χ = χ,
                        mixed=false,
                        batchsize=1000,
                        epochs=20,
                        target=Φ,
                        localnorm=false,
                        globalnorm=true)
  
else
  input_path = "data_qpt_N$(N)_noisy.h5"
  data_in,data_out,Λ = loadtrainingdataQPT(input_path; ismpo=true)
  χ = maxlinkdim(Λ)
  
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
