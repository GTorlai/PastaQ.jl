using PastaQ

N = 3
nshots = 1000

ghz = qubits(N)

applygate!(ghz,"H",1)
applygate!(ghz,"Cx",[1,2])
applygate!(ghz,"Cx",[2,3])

traindata = measure(ghz,nshots)

χ = 2
qst = QST(N=N,χ=χ)
opt = Optimizer(η = 0.01)
statetomography(qst,opt,
                data = traindata,
                batchsize=500,
                epochs=200,
                targetpsi=ghz,
                localnorm=true)

