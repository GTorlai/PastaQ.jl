using PastaQ
using Random
using ITensors

Random.seed!(1234)
N = 100
input_path = "data_ising_N$(N).h5"
data,ψ_target = loadtrainingdataQST(input_path)
χ = 2*maxlinkdim(ψ_target)

#data = data[1:10000,:]

ψ0 = initializeQST(N,χ,σ=0.1)
opt = SGD(η = 0.1)

println("Training...")
ψ = statetomography(ψ0,opt,
                    data = data,
                    batchsize=1000,
                    epochs=1000,
                    target=ψ_target,
                    localnorm=true,
                    globalnorm=false)
