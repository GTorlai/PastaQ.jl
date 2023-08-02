setup(optimizer, model::LPDO) = Optimisers.setup(optimizer, getparameters(model))

setup(optimizer, model::MPS) = PastaQ.setup(optimizer, LPDO(model))

setup(optimizer, model::MPO) = PastaQ.setup(optimizer, LPDO(model))

"""
    update!(model, grads, optimizer)

Update a tensor network model
"""
function PastaQ.update!(model, grads, st)
  L = deepcopy(model)
  θ, ∇ = getparameters(L; ∇=copy(grads))
  st, θ′ = Optimisers.update(st, θ, ∇)

  setparameters!(L, θ′)
  model.X[:] = L.X
  return model
end

function getparameters(L::LPDO; ∇=nothing)
  X = L.X
  N = length(X)
  θ = Complex{Float64}[]
  λ = Complex{Float64}[]
  for j in 1:N
    append!(θ, vec(ITensors.array(X[j])))
    !isnothing(∇) && append!(λ, vec(ITensors.array(∇[j])))
  end
  !isnothing(∇) && return θ, λ
  return θ
end

function setparameters!(L::LPDO, θ::Vector)
  X = L.X
  Y = copy(X)
  N = length(X)

  cnt = 1
  for j in 1:N
    d = dims(X[j])
    tot_d = prod(d)
    ϑ = θ[cnt:(cnt + tot_d - 1)]
    Y[j] = ITensors.itensor(reshape(ϑ, d), inds(X[j])...)
    cnt += tot_d
  end
  return X[:] = Y
end
