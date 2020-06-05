using ITensors
using PastaQ

i = Index(2)
#(dim=2|id=445)

X = quantumgates["X"](i)
#ITensor ord=2 (dim=2|id=445)' (dim=2|id=445)
#NDTensors.Dense{Float64,Array{Float64,1}}

fullmatrix(X)
#2×2 Array{Float64,2}:
# 0.0  1.0
# 1.0  0.0

R = quantumgates["Rn"](i,θ=0.1,ϕ=0.2,λ=0.3)
#ITensor ord=2 (dim=2|id=445)' (dim=2|id=445)
#NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}}

fullmatrix(R)
#2×2 Array{Complex{Float64},2}:
#   0.99875+0.0im         -0.0477469-0.0147699im
# 0.0489829+0.00992933im    0.876486+0.478826im

j = Index(2)
CX = quantumgates["Cx"](i,j)
#ITensor ord=4 (dim=2|id=445)' (dim=2|id=217)' (dim=2|id=445) (dim=2|id=217)
#NDTensors.Dense{Float64,Array{Float64,1}}

fullmatrix(CX)
#4×4 Array{Float64,2}:
# 1.0  0.0  0.0  0.0
# 0.0  1.0  0.0  0.0
# 0.0  0.0  0.0  1.0
# 0.0  0.0  1.0  0.0
