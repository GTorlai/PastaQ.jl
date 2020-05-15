include("qpt.jl")
export Optimizer
export UpdateSGD

struct Optimizer
  lr::Float32
end

function UpdateSGD(optimizer::Optimizer,qpt::QPT,grads)
  for j in 1:qpt.N
    qpt.mpo[j] = qpt.mpo[j] - optimizer.lr * grads[j]
  end
end

