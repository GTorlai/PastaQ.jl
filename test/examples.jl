using PastaQ
using ITensors
using Random
using HDF5

Random.seed!(1234)
N = 4   # Number of qubits
depth=4 # Depth of the quantum circuit
gates = randomcircuit(N,depth)

ψ = runcircuit(N,gates)
U = runcircuit(N,gates;process=true)
ρ = runcircuit(N,gates;noise="AD",γ=0.01)
Λ = runcircuit(N,gates;process=true,noise="AD",γ=0.01)

Random.seed!(1234)
nshots = 1000
ψ,data = getsamples(N,gates,nshots;return_state=true)
savedata(ψ,data,"../examples/data/qst_circuit_test.h5")

ρ,data = getsamples(N,gates,nshots;
                      noise="AD",γ=0.01,
                      return_state=true)
savedata(ρ,data,"../examples/data/qst_circuit_noisy_test.h5")

U,data_in,data_out=getsamples(N,gates,nshots;
                                  process=true,
                                  return_state=true)
savedata(U,data_in,data_out,"../examples/data/qpt_circuit_test.h5")

Λ,data_in,data_out=getsamples(N,gates,nshots;
                                  process=true,
                                  noise="AD",γ=0.01,
                                  return_state=true)
savedata(Λ,data_in,data_out,"../examples/data/qpt_circuit_noisy_test.h5")


Random.seed!(1234)
Ψ,data = loaddata("../examples/data/qst_circuit_test.h5")
N = length(Ψ)     # Number of qubits
χ = maxlinkdim(Ψ) # Bond dimension of variational MPS
ψ0 = randomstate(Ψ;χ=χ,σ=0.1)
opt = SGD(η = 0.01)
ψ = tomography(ψ0,data,opt;
               batchsize=100,
               epochs=2,
               target=Ψ);

ϱ,data = loaddata("../examples/data/qst_circuit_noisy_test.h5")
N = length(ϱ)     # Number of qubits
χ = maxlinkdim(ϱ) # Bond dimension of variational LPDO
ξ = 2             # Kraus dimension of variational LPDO
ρ0 = randomstate(ϱ;mixed=true,χ=χ,ξ=ξ,σ=0.1)
opt = SGD(η = 0.01)
ρ = tomography(ρ0,data,opt;
               batchsize=100,
               epochs=2,
               target=ϱ);

Random.seed!(1234)
U,data_in,data_out = loaddata("../examples/data/qpt_circuit_test.h5";process=true)
N = length(U)     # Number of qubits
χ = maxlinkdim(U) # Bond dimension of variational MPS
opt = SGD(η = 0.1)
V0 = randomprocess(U;mixed=false,χ=χ)
@show V0
V = tomography(V0,data_in,data_out,opt;
               batchsize=100,
               epochs=2,
               target=U)

# Noisy circuit
Random.seed!(1234)
ϱ,data_in,data_out = loaddata("../examples/data/qpt_circuit_noisy_test.h5";process=true)
N = length(ϱ)
χ = 8
ξ = 2
Λ0 = randomprocess(ϱ;mixed=true,χ=χ,ξ=ξ,σ=0.1)
opt = SGD(η = 0.1)
Λ = tomography(Λ0,data_in,data_out,opt;
               mixed=true,
               batchsize=100,
               epochs=2,
               target=ϱ);
@show Λ
