using LinearAlgebra
using ITensors

struct QPT
  prep::String
  povm::String
  K::Int
  #M::Vector{ITensor}
end

function QPT(;prep::String,povm::String)
  # Set the input states
  if prep == "Pauli6"
    K = 6
    P = ITensor[]
    i = Index(2)
    mat = [1 0; 0 0]
    push!(P,ITensor(mat,i,i'))
    mat = [0 0; 0 1]
    push!(P,ITensor(mat,i,i'))
    mat = 1/2. * [1 1; 1 1]
    push!(P,ITensor(mat,i,i'))
    mat = 1/2. * [1 -1; -1 1]
    push!(P,ITensor(mat,i,i'))
    mat = 1/2. * [1 1im; -1im 1]
    push!(P,ITensor(mat,i,i'))
    mat = 1/2. * [1 -1im; 1im 1]
    push!(P,ITensor(mat,i,i'))
  else
    error("Only Pauli6 preparation is implemented")
  end
  
  # Set the POVM elements
  if povm == "Pauli6"
    K = 6
    M = ITensor[]
    i = Index(2)
    mat = 1/3. * [1 0; 0 0]
    push!(M,ITensor(mat,i,i'))
    mat = 1/3. * [0 0; 0 1]
    push!(M,ITensor(mat,i,i'))
    mat = 1/6. * [1 1; 1 1]
    push!(M,ITensor(mat,i,i'))
    mat = 1/6. * [1 -1; -1 1]
    push!(M,ITensor(mat,i,i'))
    mat = 1/6. * [1 -1im; 1im 1]
    push!(M,ITensor(mat,i,i'))
    mat = 1/6. * [1 1im; -1im 1]
    push!(M,ITensor(mat,i,i'))
    @assert sum(M) ≈ delta(i,i') atol=1e-8
  else
    error("Only Pauli6 POVMs are implemented")
  end
  
  return QPT(prep,povm,K)

end




