using PastaQ
using Random
using ITensors

Random.seed!(1234)
N = 4
input_path = "../../data/qst/data_LiH_N$(N).h5"
data,target = loadtrainingdataQST(input_path)

χ = maxlinkdim(target)
psi0 = initializeQST(N,χ,σ=0.1)
opt = Sgd(η = 0.1)
println("Training...")
psi = statetomography(psi0,opt,
                      data=data,
                      batchsize=1000,
                      epochs=1000,
                      target=target,
                      localnorm=true)
