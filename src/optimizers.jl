PastaQ.state(optimizer, model::LPDO) = 
  optimizerstate(optimizer, getparameters(model))

PastaQ.state(optimizer, model::MPS) = 
  PastaQ.state(optimizer, LPDO(model))

PastaQ.state(optimizer, model::MPO) = 
  PastaQ.state(optimizer, LPDO(model))


"""
    update!(model, grads, optimizer)

Update a tensor network model
"""
function PastaQ.update!(model, grads, optimizer)
  L = deepcopy(model)
  opt, st = optimizer
  θ, ∇ = getparameters(L; ∇ = copy(grads))
  st, θ′ = Optimisers.update(opt, st, θ, ∇)
  
  setparameters!(L, θ′)
  model.X[:] = L.X
  return model
end

function gettensorindices(M::MPS, site::Int)
  if haschoitags(M)
    if site == 1
      i = firstind(M[site], tags="Input") 
      o = firstind(M[site], tags="Output") 
      l = commonind(M[1],M[2]) 
      return (i,o,l)
    elseif site == length(M)
      i = firstind(M[site], tags="Input") 
      o = firstind(M[site], tags="Output") 
      l = commonind(M[site],M[site-1])
      return (i,o,l)
    else
      i = firstind(M[site], tags="Input") 
      o = firstind(M[site], tags="Output") 
      l1 = commonind(M[site],M[site-1])
      l2 = commonind(M[site],M[site+1])
      return (i,o,l1,l2)
    end
  else
    if site == 1
      s = firstind(M[site], tags="Site") 
      l = commonind(M[1],M[2]) 
      return (s,l)
    elseif site == length(M)
      s = firstind(M[site], tags="Site") 
      l = commonind(M[site],M[site-1])
      return (s,l)
    else
      s  = firstind(M[site], tags="Site") 
      l1 = commonind(M[site],M[site-1])
      l2 = commonind(M[site],M[site+1])
      return (s,l1,l2)
    end
  end
end

function gettensorindices(M::MPO, site::Int)
  if haschoitags(M)
    if site == 1
      i = firstind(M[site], tags="Input", plev = 0) 
      o = firstind(M[site], tags="Output", plev = 0) 
      ξ = firstind(M[site], tags="Purifier") 
      l = commonind(M[1],M[2]) 
      return (i,o,l,ξ)
    elseif site == length(M)
      i = firstind(M[site], tags="Input", plev = 0) 
      o = firstind(M[site], tags="Output", plev = 0) 
      ξ = firstind(M[site], tags="Purifier") 
      l = commonind(M[site],M[site-1])
      return (i,o,l,ξ)
    else
      i = firstind(M[site], tags="Input", plev = 0) 
      o = firstind(M[site], tags="Output", plev = 0) 
      ξ = firstind(M[site], tags="Purifier") 
      l1 = commonind(M[site],M[site-1])
      l2 = commonind(M[site],M[site+1])
      return (i,o,l1,l2,ξ)
    end
  else
    if site == 1
      s = firstind(M[site], tags="Site") 
      ξ = firstind(M[site], tags="Purifier") 
      l = commonind(M[1],M[2]) 
      return (s,l,ξ)
    elseif site == length(M)
      s = firstind(M[site], tags="Site") 
      ξ = firstind(M[site], tags="Purifier") 
      l = commonind(M[site],M[site-1])
      return (s,l,ξ)
    else
      s = firstind(M[site], tags="Site") 
      ξ = firstind(M[site], tags="Purifier") 
      l1 = commonind(M[site],M[site-1])
      l2 = commonind(M[site],M[site+1])
      return (s,l1,l2,ξ)
    end
  end
end

function getparameters(L::LPDO; ∇ = nothing)
  X = L.X
  N = length(X)
  θ = Complex[]
  λ = Complex[]
  # either state or gradients
  for j in 1:N
    legs = gettensorindices(X, j)
    basis = [collect(1:x) for x in dim.(legs)]
    basis = vec(Iterators.product(basis...)|>collect)
    for μ in basis
      tels = legs .=> μ
      push!(θ, X[j][tels...])
      !isnothing(∇) && push!(λ, ∇[j][tels...])
    end
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
    legs = gettensorindices(X, j)
    basis = [collect(1:x) for x in dim.(legs)]
    basis = vec(Iterators.product(basis...)|>collect)
    for μ in basis
      tels = legs .=> μ
      Y[j][tels...] = θ[cnt]
      cnt += 1
    end
  end
  X[:] = Y
  return X
end


