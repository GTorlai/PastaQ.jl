using PastaQ
using Random

Random.seed!(123456)
N = 3
input_path = "../../data/qst/data_ghz_N$(N).h5"
samples,bases,target = loadtrainingdataQST(input_path)

χ = 2
psi = initializeQST(N,χ)
opt = Sgd(η = 0.01)
statetomography!(psi,opt,
                samples = samples,
                bases = bases,
                batchsize=500,
                epochs=1000,
                target=target,
                localnorm=true)

