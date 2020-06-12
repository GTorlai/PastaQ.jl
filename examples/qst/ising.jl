using PastaQ
using Random
using ITensors

Random.seed!(1234)
N = 20
input_path = "../../data/qst/data_ising_N$(N).h5"
samples,bases,target = loadtrainingdataQST(input_path)

data = Matrix{String}(undef, size(samples)[1],N)
for n in 1:size(samples)[1]
  data[n,:] = convertdata(samples[n,:],bases[n,:])
end

χ = maxlinkdim(target)
psi = initializeQST(N,χ,σ=0.1)
opt = Sgd(η = 0.1)
println("Training...")
@time statetomography!(psi,opt,
                 samples = samples,
                 bases = bases,
                 batchsize=1000,
                 epochs=5,
                 target=target,
                 localnorm=true)
