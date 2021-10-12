using PastaQ
using ITensors
using Test
using Random
using Printf
import PastaQ: gate


function numgradpars(circuit::Vector{<:Any}, costfunction::MPO; ϵ = 1e-7)
  N = length(costfunction)
  depth = length(circuit)
  numgradients = []
  cmap = PastaQ.circuitmap(circuit)
  
  for g in cmap
    gatename, support, params = circuit[g]
    par_ids = keys(params)
    ∇g = []
    for par_id in par_ids
      if par_id != :∇
        angle = params[par_id]
        angle = angle + ϵ
        circuit[g] = Base.setindex(circuit[g],Base.setindex(circuit[g][3],angle,par_id),3)
        Ep = PastaQ.loss(circuit, costfunction)
        angle = angle - ϵ
        circuit[g] = Base.setindex(circuit[g],Base.setindex(circuit[g][3],angle,par_id),3)
        Em = PastaQ.loss(circuit, costfunction)
        numgrad = (Ep-Em)/ϵ
        push!(∇g, par_id => numgrad)
      end
    end
    push!(numgradients, ∇g)
  end

  #for d in 1:depth
  #  for i in 1:length(circuit[d])
  #    if length(circuit[d][i]) == 3
  #      par_ids =  keys(circuit[d][i][3])
  #      for par_id in par_ids
  #        if par_id != :∇
  #          angle = circuit[d][i][3][par_id]
  #          
  #          angle = angle + ϵ
  #          circuit[d][i] = Base.setindex(circuit[d][i],Base.setindex(circuit[d][i][3],angle,par_id),3) 
  #          Ep = PastaQ.loss(circuit, costfunction)   
  #          
  #          angle = angle - ϵ
  #          circuit[d][i] = Base.setindex(circuit[d][i],Base.setindex(circuit[d][i][3],angle,par_id),3)
  #          Em = PastaQ.loss(circuit, costfunction)   
  #          numgrad = (Ep-Em)/ϵ
  #          push!(numgradients, numgrad)
  #        end
  #      end
  #    end
  #  end
  #end
  return numgradients
end


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

#
#
#@testset "VQE-parameters gradients" begin
#  Random.seed!(1234)
#  N = 4
#  depth = 2
#  H,_ = isingmodel(N)
#  circuit = variationalcircuit(randomcircuit(N, depth; twoqubitgates = "CX", onequbitgates = ["Rn"], layered = false))
#  ∇num = numgradpars(circuit, H)
#  L, _, ∇ = PastaQ.gradients(circuit, H) 
#  @test L ≈ PastaQ.loss(circuit, H)
#  for i in 1:length(∇num)
#    for (k,grad) in enumerate(∇num[i])
#      @test last(grad) ≈ last(∇[i][k]) atol = 1e-6
#    end
#  end
#end
#
#@testset "VQE: update angles" begin
#  Random.seed!(1234)
#  N = 4
#  depth = 3
#  H,_ = isingmodel(N)
#  circuit0 = variationalcircuit(randomcircuit(N, depth; twoqubitgates = "CX", onequbitgates = ["Rn"], layered = false))
#  circuit = copy(circuit0)
#  cmap = PastaQ.circuitmap(circuit)
#  trainpars = PastaQ.trainableparameters(circuit)
#  θ0 = PastaQ._getparameters(circuit, cmap, trainpars)
#  npars  = 3 * 4 * 3
#  θ1 = rand(npars)
#  PastaQ._setparameters!(circuit, θ1, cmap, trainpars)
#  @test θ1 ≈ PastaQ._getparameters(circuit, cmap, trainpars) 
#
#  layer = Tuple[("Rx", 1, (θ = 0.1, ∇ = true)),("Rx", 2, (θ = 0.1, ∇ = true)),("Rx", 3, (θ = 0.1, )),("Rx", 4, (θ = 0.1, ))]
#  circuit = push!(circuit,layer...)
#  cmap = PastaQ.circuitmap(circuit)
#  trainpars = PastaQ.trainableparameters(circuit)
#  # update raw
#  θ0 = PastaQ._getparameters(circuit, cmap, trainpars)
#  npars += 2
#  θ1 = rand(npars)
#  PastaQ._setparameters!(circuit, θ1, cmap, trainpars)
#  @test θ1 ≈ PastaQ._getparameters(circuit, cmap, trainpars) 
#  
#end
