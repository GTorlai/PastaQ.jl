using PastaQ
using ITensors

N = 50
h = 1.0
H,ψ = transversefieldising(N,h,hamiltonian=true)
E = inner(ψ,H,ψ)
println("GS : <H> = $E  (χ = $(maxlinkdim(ψ)))")

τ = 0.02 
δ = 1.0
for b in 1:100
  β = b * δ
  depth = β ÷ τ

  ρ0 = circuit(firstsiteinds(H))
  orthogonalize!(ρ0,1)
  gate_tensors = ITensor[]
  # layers
  for d in 1:depth
    # ising interaction
    for j in 1:N-1;
      z1 = gate(ρ0,"Z",j)
      z2 = gate(ρ0,"Z",j+1)
      zz = z1 * z2
      g  = exp(τ * zz) 
      push!(gate_tensors,g)
    end 
    # transverse field
    for j in 1:N 
      x = gate(ρ0,"X",j)
      g = exp(τ * h * x)
      push!(gate_tensors,g)
    end 
  end 

  ρ = runcircuit(ρ0,gate_tensors,state_evolution=false,
                 cutoff=1e-9,maxdim=1000)
  Z = trace_mpo(ρ)
  for j in 1:N 
    ρ[j] = ρ[j] / Z^(1.0/N)
  end 
  
  E_th = trace_mpo(*(ρ,prime(H),method="densitymatrix",cutoff=1e-10))
  println("β = $β : <H> = $E_th  (χ = $(maxlinkdim(ρ)))")
end


""" SINGLE RUN """
#N = 10
#h = 1.0
#β = 1.0
#τ = 0.1
#depth = β ÷ τ 
#
#ρ0 = circuit(N)
#orthogonalize!(ρ0,1)
#
#gate_tensors = ITensor[]
#
## layers
#for d in 1:depth
#  # ising interaction
#  for j in 1:N-1;
#    z1 = gate(ρ0,"Z",j)
#    z2 = gate(ρ0,"Z",j+1)
#    zz = z1 * z2
#    g  = exp(τ * zz) 
#    push!(gate_tensors,g)
#  end 
#  # transverse field
#  for j in 1:N 
#    x = gate(ρ0,"X",j)
#    g = exp(τ * h * x)
#    push!(gate_tensors,g)
#  end 
#end 
#
## Run circuit
#ρ = runcircuit(ρ0,gate_tensors,state_evolution=false)
#
## Normalize the density matrix
#Z = trace_mpo(ρ)
#for j in 1:N 
#  ρ[j] = ρ[j] / Z^(1.0/N)
#end 


