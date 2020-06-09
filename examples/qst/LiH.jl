using PastaQ
using Random
using ITensors

Random.seed!(1234)
N = 4
input_path = "../../data/qst/data_LiH_N$(N).h5"
samples,bases,target = loadtrainingdataQST(input_path)

samples = samples[1:5000,:]
bases  = bases[1:5000,:]

χ = maxlinkdim(target)
psi = initializeQST(N,χ,σ=0.1)
opt = Sgd(η = 0.1)
println("Training...")
statetomography!(psi,opt,
                 samples = samples,
                 bases = bases,
                 batchsize=1000,
                 epochs=1000,
                 target=target,
                 localnorm=true)
