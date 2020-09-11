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

