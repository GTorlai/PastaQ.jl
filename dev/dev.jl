using LinearAlgebra
using ITensors
using PyCall
pickle = pyimport("pickle");
include("test.jl")
include("qpt_unitary.jl")
#include("optimizer.jl")

" RANDOM INITIALIZATION "
N    = 3
chi  = 2
K    = 6
d    = 2
ns   = 10000
seed =1234
learning_rate = 0.01

training_data = rand(1:K,ns,2*N)
training_data .+= 1

qpt = QPT(
  povm="Pauli6",
  prep="Pauli6",
  N=N,
  chi=chi,
  seed=seed)

" USE MPO GENERATTED FROM TF VERSION FOR BENCHMARKING "
path = string("testdata_N",N,"_chi",chi,".pickle")
f_in = open(path)
testdata = pickle.load(f_in);
training_data = testdata["dataset"]["samples"];
training_data .+= 1
path = string("tfdata_N",N,"_chi",chi,".pickle")
f_in = open(path)
tfdata = pickle.load(f_in);
mpo_tf = tfdata["mps"];
SetMPO(qpt,mpo_tf)

#normalization = Normalization(qpt.mpo,choi=true)
#println(normalization)
batch = training_data[1:10,:]
#KL_loss = Loss(qpt,batch)
#println(KL_loss)

target_mpo = SetTargetMPO(qpt,testdata["choi"]["mps"])

optimizer = Optimizer(learning_rate)
RunTomography(qpt,optimizer,training_data,target_mpo)
#@show target_mpo[1]
#println(QuantumProcessFidelity(qpt,target_mpo))
#println(Normalization(target_mpo,choi=false))
#println(Normalization(target_mpo,choi=true))
#grad_Z,_ = GradientLogZ(qpt)
#num_grad_Z = NumericalGradientLogZ(qpt.mpo,accuracy=1e-8)
#CheckGradients(grad_Z,num_grad_Z)
#print("\n\n")
#num_grad_P = NumericalGradientKL(qpt,batch,accuracy=1e-8)
#grad_P,_ = GradientKL(qpt,batch)
#CheckGradients(grad_P,num_grad_P)


#optimizer = Optimizer(learning_rate)
#
#@show qpt.mpo[1]
#
#grad_P = GradientKL(qpt,batch)
#
#UpdateSGD(optimizer,qpt,grad_P)
#
#@show qpt.mpo[1]






#
