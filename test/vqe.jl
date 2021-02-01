using PastaQ
using ITensors
using Test
using Random
using Printf

function isingmodel(N::Int)
  # generate the hamiltonian MPO
  sites = siteinds("Qubit",N)
  ampo = AutoMPO()
  
  # loop over the pauli operators
  for j in 1:N-1
    ampo .+= -1.0,"Z",j,"Z",j+1 
    ampo .+= -1.0,"X",j
  end
  ampo .+= -1.0,"X",N
  H = MPO(ampo,sites)
  
  # find ground state with DMRG
  mps = randomMPS(sites)
  sweeps = Sweeps(10)
  maxdim!(sweeps, 10,20,30,50,100)
  cutoff!(sweeps, 1E-10)
  E0, mps = dmrg(H, mps, sweeps, outputlevel = 0);
  #@printf("\nGround state energy: %.10f\n\n",E0)
  return H,mps  
end

@testset "gate derivatives" begin
  ϵ = 1e-8
  θ = 0.1
  ϕ = 0.2
  λ = 0.3

  Gp = gate("Rx"; θ = θ + ϵ)
  Gm = gate("Rx"; θ = θ - ϵ)
  ∇num = (Gp-Gm)/(2*ϵ)
  ∇ = PastaQ.gradient("Rx"; θ = θ)[:θ]
  @test ∇ ≈ ∇num

  Gp = gate("Ry"; θ = θ + ϵ)
  Gm = gate("Ry"; θ = θ - ϵ)
  ∇num = (Gp-Gm)/(2*ϵ)
  ∇ = PastaQ.gradient("Ry"; θ = θ)[:θ]
  @test ∇ ≈ ∇num

  Gp = gate("Rz"; ϕ = ϕ + ϵ)
  Gm = gate("Rz"; ϕ = ϕ - ϵ)
  ∇num = (Gp-Gm)/(2*ϵ)
  ∇ = PastaQ.gradient("Rz"; ϕ = ϕ)[:ϕ]
  @test ∇ ≈ ∇num 
  
  Gp = gate("Rn"; θ = θ + ϵ, ϕ = ϕ, λ = λ)
  Gm = gate("Rn"; θ = θ - ϵ, ϕ = ϕ, λ = λ)
  ∇num = (Gp-Gm)/(2*ϵ)
  ∇ = PastaQ.gradient("Rn"; θ = θ, ϕ = ϕ, λ = λ)[:θ]
  @test ∇ ≈ ∇num
  
  Gp = gate("Rn"; θ = θ, ϕ = ϕ + ϵ, λ = λ)
  Gm = gate("Rn"; θ = θ, ϕ = ϕ - ϵ, λ = λ)
  ∇num = (Gp-Gm)/(2*ϵ)
  ∇ = PastaQ.gradient("Rn"; θ = θ, ϕ = ϕ, λ = λ)[:ϕ]
  @test ∇ ≈ ∇num
  
  Gp = gate("Rn"; θ = θ, ϕ = ϕ, λ = λ + ϵ)
  Gm = gate("Rn"; θ = θ, ϕ = ϕ, λ = λ - ϵ)
  ∇num = (Gp-Gm)/(2*ϵ)
  ∇ = PastaQ.gradient("Rn"; θ = θ, ϕ = ϕ, λ = λ)[:λ]
  @test ∇ ≈ ∇num
  
end

@testset "vqe environments" begin
  N = 5
  depth = 3

  H, _ = isingmodel(N)
  Hmat = array(H)
  circuit = randomcircuit(N, depth; twoqubitgates = "CX", onequbitgates = ["Rn"])
  vqe = VQE(H, circuit)

  ΨL, ΨR = PastaQ.environments(vqe)
  # test L
  ψ = array(qubits(N))
  @test array(ΨL[1]) ≈ ψ
  for d in 1:depth-1
    ψ = array(runcircuit(vqe.circuit[d]; process = true)) * ψ
    @test array(ΨL[d+1]) ≈ ψ
  end
  
  ψ = array(runcircuit(circuit))
  ψ = Hmat * ψ
  @test ψ ≈ array(ΨR[end])
  for d in reverse(2:depth)
    U = array(runcircuit(dag(vqe.circuit[d]); process = true))
    ψ = U * ψ
    @test ψ ≈ array(ΨR[d-1])
  end
end

istrainable(g::Tuple) = length(g) == 3

function VQEenergy(H::MPO, gates::Union{Vector{<:Vector{<:Tuple}},Vector{<:ITensor}})
  ψθ = runcircuit(qubits(H), gates)
  E = inner(ψθ, H, ψθ)
  @assert(imag(E)<1e-7)  
  return real(E)
end

function numgradpars(vqe::PastaQ.VariationalQuantumEigensolver; ϵ = 1e-7)
  N = length(vqe.Hamiltonian)
  depth = length(vqe.circuit)
  numgradients = []
  for d in 1:depth
    for i in 1:length(vqe.circuit[d])
      if istrainable(vqe.circuit[d][i])
        par_ids =  keys(vqe.circuit[d][i][3])
        for par_id in par_ids
          angle = vqe.circuit[d][i][3][par_id]
          
          angle = angle + ϵ
          vqe.circuit[d][i] = Base.setindex(vqe.circuit[d][i],Base.setindex(vqe.circuit[d][i][3],angle,par_id),3) 
          Ep = VQEenergy(vqe.Hamiltonian,vqe.circuit)   
          
          angle = angle - ϵ
          vqe.circuit[d][i] = Base.setindex(vqe.circuit[d][i],Base.setindex(vqe.circuit[d][i][3],angle,par_id),3)
          Em = VQEenergy(vqe.Hamiltonian,vqe.circuit)
          numgrad = (Ep-Em)/ϵ
          push!(numgradients, numgrad)
        end
      end
    end
  end
  return numgradients
end
@testset "VQE-parameters gradients" begin
  Random.seed!(1234)
  N = 4
  depth = 4
  H,_ = isingmodel(N)
  circuit = randomcircuit(N, depth; twoqubitgates = "CX", onequbitgates = ["Rn"])
  vqe = VQE(H,circuit) 
  ∇num = numgradpars(vqe)
  ∇ = PastaQ.gradients(vqe) 
  counter = 1 
  
  for i in 1:length(∇num)
    @test ∇num[i] ≈ ∇[i] atol = 1e-6
  end
  #for d in 1:length(∇)
  #  for i in 1:length(∇[d])
  #    @test ∇num[counter] ≈ ∇[d][i] atol = 1e-6
  #    counter += 1
  #  end
  #end
end

@testset "VQE: update angles" begin
  Random.seed!(1234)
  N = 4
  depth = 3
  H,_ = isingmodel(N)
  circuit = randomcircuit(N, depth; twoqubitgates = "CX", onequbitgates = ["Rn"])
  vqe0 = VQE(H,circuit) 
  vqe = copy(vqe0)
  
  θ0 = PastaQ.getparameters(vqe)
  npars  = 3 * 4 * 3
  θ1 = rand(npars)
  PastaQ.updateangles!(vqe,θ1)
  @test θ1 ≈ PastaQ.getparameters(vqe) 

  layer = layer = Tuple[("Rx", 1, (θ = 0.1,)),("Rx", 2, (θ = 0.1,)),("Rx", 3, (θ = 0.1, nograd = true)),("Rx", 4, (θ = 0.1, nograd = true))]
  newcircuit = push!(circuit,layer)
  vqe = VQE(H, circuit)
  θ0 = PastaQ.getparameters(vqe)
  npars += 2
  θ1 = rand(npars)
  PastaQ.updateangles!(vqe,θ1)
  @test θ1 ≈ PastaQ.getparameters(vqe) 

end



#  ∇num = numgradpars(vqe0)
#  ∇ = PastaQ.gradients(vqe) 
#  η = 0.01
#  PastaQ.updateangles!(vqe, ∇; η = η)
#  counter = 1 
#  for d in 1:depth
#    trainable_index = findall(x -> x==1, PastaQ.istrainable.(vqe.circuit[d]))
#    for i in 1:length(trainable_index)
#      par_id =  keys(vqe.circuit[d][trainable_index[i]][3])
#      for par in par_id
#        @test ∇[d][i] ≈ ∇num[counter] atol = 1e-6
#        old_angle = vqe0.circuit[d][trainable_index[i]][3][par]
#        X = old_angle - η * ∇num[counter]
#        @test X ≈ vqe.circuit[d][trainable_index[i]][3][par]
#        counter += 1
#      end
#    end
#  end
#end
