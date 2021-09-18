"""
    update!(model, grads, optimizer)

Update a tensor network model
"""
function update!(model, grads, optimizer)
  M0 = model.X
  N = length(M0)
  θ = [ITensors.array(M0[j]) for j in 1:N]
  ∇ = [ITensors.array(grads[j]) for j in 1:N]
  θvec = vcat([vec(θ[j]) for j in 1:N]...)
  ∇vec = vcat([vec(∇[j]) for j in 1:N]...)
           
  Flux.Optimise.update!(optimizer, θvec, ∇vec)
  
  start = 1
  for j in 1:N
    stop = start+prod(dim.(inds(model.X[j])))-1
    θ[j] = reshape(θvec[start:stop],dim.(inds(model.X[j])))
    start = stop + 1
  end
  M = [itensor(θ[j], inds(M0[j])) for j in 1:N] 
  
  M = M0 isa MPS ? MPS(M) : MPO(M)
  model.X[:] = M
  return model
end

