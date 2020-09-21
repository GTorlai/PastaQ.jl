using PastaQ
using Random
using ITensors

Random.seed!(1234)
N = 10
input_path = "data_ising_N$(N).h5"
data,ψ_target = loadtrainingdataQST(input_path)
χ = maxlinkdim(ψ_target)

#data = data[1:1000,:]

#ψ0 = initializetomography(N,χ,σ=0.1)
opt = SGD(η = 0.1)

println("Training...")
#ψ = statetomography(ψ0,data,opt;
ψ = statetomography(data,opt;
                    χ=χ,
                    mixed=false,
                    batchsize=500,
                    epochs=1000,
                    target=ψ_target,
                    localnorm=true,
                    globalnorm=false)
