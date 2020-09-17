using PastaQ
using Random
using ITensors

Random.seed!(1234)
N = 2
input_path = "data_ising_N$(N).h5"
data,ψ_target = loadtrainingdataQST(input_path)
χ = maxlinkdim(ψ_target)

#data = data[1:10000,:]

ψ0 = initializetomography(N,χ,σ=0.1)
#opt = SGD(η = 0.01)
#opt = Momentum(ψ0;η = 0.01, μ = 0.9)
opt = Adagrad(ψ0;η=0.01,ϵ=1E-8)

obs = TomographyObserver(["Z"],ψ0)
println("Training...")
#ψ = statetomography(ψ0,data,opt;
(ψ,obs) = statetomography(ψ0,data,opt;observer=obs,
                          #fout="prova.h5",
                          χ=χ,
                          mixed=false,
                          batchsize=500,
                          epochs=2,
                          target=ψ_target,
                          localnorm=true,
                          globalnorm=false)
