using ITensors
using PastaQ

N = 3
sites = siteinds("S=1/2",N)

# Input operator terms which define 
# a Hamiltonian matrix, and convert
# these terms to an MPO tensor network
# (here we make the 1D Ising model)
ampo = AutoMPO()
for j=1:N-1
  ampo .+=(-1.0,"Sz",j,"Sz",j+1)
  ampo .+=(-1.0,"Sx",j) 
end
ampo += (-1.0,"Sx",N)
H = MPO(ampo,sites)

# Create an initial random matrix product state
psi0 = randomMPS(sites)

# Plan to do 5 passes or 'sweeps' of DMRG,
# setting maximum MPS internal dimensions 
# for each sweep and maximum truncation cutoff
# used when adapting internal dimensions:
sweeps = Sweeps(5)
maxdim!(sweeps, 10,20,100,100,200)
cutoff!(sweeps, 1E-10)

# Run the DMRG algorithm, returning energy 
# (dominant eigenvalue) and optimized MPS
energy, targetpsi = dmrg(H,psi0, sweeps)
println("Final energy = $energy")

nshots = 5000
traindata = measure(targetpsi,nshots)

χ = maxlinkdim(targetpsi)

qst = QST(N=N,χ=χ,σ=0.1)

opt = Optimizer(η = 0.1)

statetomography(qst,opt,
                data = traindata,
                batchsize=500,
                epochs=10000,
                targetpsi=targetpsi,
                localnorm=false)


