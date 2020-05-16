#include("qpt.jl")
struct Sgd
  lr::Float64
end

#function Update(opt::Sgd,qpt::QPT,grads)
#  for j in 1:qpt.N
#    qpt.mpo[j] = qpt.mpo[j] - optimizer.lr * grads[j]
#  end
#end

