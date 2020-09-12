using ITensors
using PastaQ
using Random
using HDF5

Random.seed!(1234)
N = 20
input_path = "data_qpt_N$(N)_unitary.h5"
data_in,data_out,Φ = loadtrainingdataQPT(input_path)
χ = maxlinkdim(Φ)

opt = SGD(η = 0.1)

Ψ = processtomography(data_in,data_out,opt;
                      χ = χ,
                      mixed=false,
                      batchsize=1000,
                      epochs=1000,
                      target=Φ,
                      localnorm=true,
                      globalnorm=false)


#N = 2
#input_path = "data_qpt_N$(N)_noisy.h5"
#data_in,data_out,Λ = loadtrainingdataQPT(input_path; ismpo=true)
#χ = maxlinkdim(Φ)
#
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
