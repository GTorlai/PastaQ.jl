using ITensors
using PastaQ

psi = qubits(3)
#MPS
#[1] IndexSet{2} (dim=2|id=89|"Site,n=1") (dim=1|id=504|"Link,l=1") 
#[2] IndexSet{3} (dim=1|id=504|"Link,l=1") (dim=2|id=314|"Site,n=2") (dim=1|id=243|"Link,l=2") 
#[3] IndexSet{2} (dim=1|id=243|"Link,l=2") (dim=2|id=678|"Site,n=3") 

""" BUILD QUANTUM GATES """
gate = makegate(psi,"X",1)
#ITensor ord=2 (dim=2|id=89|"Site,n=1")' (dim=2|id=89|"Site,n=1")
#NDTensors.Dense{Float64,Array{Float64,1}}

gate = makegate(psi,"Cx",[1,2])
#ITensor ord=4 (dim=2|id=89|"Site,n=1")' (dim=2|id=314|"Site,n=2")' (dim=2|id=89|"Site,n=1") (dim=2|id=314|"Site,n=2")
#NDTensors.Dense{Float64,Array{Float64,1}}

gateid = (gate = "X", site = 1)
gate = makegate(psi,gateid)
#ITensor ord=2 (dim=2|id=89|"Site,n=1")' (dim=2|id=89|"Site,n=1")
#NDTensors.Dense{Float64,Array{Float64,1}}

gateid = (gate = "Rn", site = 1,params=(θ=0.1,ϕ=0.2,λ=0.3))
gate = makegate(psi,gateid)
#ITensor ord=2 (dim=2|id=89|"Site,n=1")' (dim=2|id=89|"Site,n=1")
#NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}}

""" APPLY QUANTUM GATES """
psi = qubits(2)
#MPS
#[1] IndexSet{2} (dim=2|id=88|"Site,n=1") (dim=1|id=251|"Link,l=1") 
#[2] IndexSet{2} (dim=1|id=251|"Link,l=1") (dim=2|id=304|"Site,n=2") 

fullvector(psi)
#4-element Array{Float64,1}:
# 1.0
# 0.0
# 0.0
# 0.0

gateid = (gate = "X", site = 1)
gate = makegate(psi,gateid)
#ITensor ord=2 (dim=2|id=88|"Site,n=1")' (dim=2|id=88|"Site,n=1")
#NDTensors.Dense{Float64,Array{Float64,1}}

applygate!(psi,gate)
#ITensor ord=2 (dim=2|id=88|"Site,n=1") (dim=1|id=251|"Link,l=1")
#NDTensors.Dense{Float64,Array{Float64,1}}

fullvector(psi)
#4-element Array{Float64,1}:
# 0.0
# 0.0
# 1.0
# 0.0

applygate!(psi,"Rn",1,θ=0.1,ϕ=0.2,λ=0.3)
fullvector(psi)
#4-element Array{Complex{Float64},1}:
# -0.04774692410046421 - 0.014769854431632931im
#                  0.0 - 0.0im
#   0.8764858122060915 + 0.4788263815209447im
#                  0.0 + 0.0im
