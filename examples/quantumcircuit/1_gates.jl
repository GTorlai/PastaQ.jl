using ITensors
using PastaQ

i = Index(2)
#(dim=2|id=445)

X = gate("X", i)
#ITensor ord=2 (dim=2|id=445)' (dim=2|id=445)
#NDTensors.Dense{Float64,Array{Float64,1}}

@show matrix(X) ==
  [0.0  1.0
   1.0  0.0]

R = gate("Rn",i,θ=0.1,ϕ=0.2,λ=0.3)
#ITensor ord=2 (dim=2|id=445)' (dim=2|id=445)
#NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}}

@show matrix(R) ≈
  [0.9987502603949663 + 0.0im                   -0.04774692410046421 - 0.014769854431632931im
   0.04898291339046185 + 0.009929328112698753im   0.8764858122060915 + 0.4788263815209447im]

j = Index(2)
CX = gate("CX",i,j)
#ITensor ord=4 (dim=2|id=445)' (dim=2|id=217)' (dim=2|id=445) (dim=2|id=217)
#NDTensors.Dense{Float64,Array{Float64,1}}

C = combiner(j, i)
@show matrix(CX * C' * C) ==
  [1.0  0.0  0.0  0.0
   0.0  1.0  0.0  0.0
   0.0  0.0  0.0  1.0
   0.0  0.0  1.0  0.0]