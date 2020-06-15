using PastaQ
using Random

Random.seed!(123456)
N = 20
input_path = "../../data/qst/data_ghz_N$(N).h5"
data,target = loadtrainingdataQST(input_path)

χ = 2
psi0 = initializeQST(N,χ)
opt = Sgd(η = 0.1)
psi = statetomography(psi0, opt,
                      data = data,
                      batchsize=500,
                      epochs=1000,
                      target=target,
                      localnorm=true)

