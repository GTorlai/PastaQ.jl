using PastaQ
using Random
using ITensors

Random.seed!(1234)
N = 20
input_path = "../../data/qst/data_ising_N$(N).h5"
data,target = loadtrainingdataQST(input_path)

χ = maxlinkdim(target)
ξ = 4
lpdo0 = initializeQST(N,χ,ξ,σ=0.1)
opt = SGD(η = 0.1)
println("Training...")
lpdo = statetomography(lpdo0,opt,
                       data = data,
                       batchsize=1000,
                       epochs=5,
                       target=target,
                       localnorm=true,
                       globalnorm=false)
