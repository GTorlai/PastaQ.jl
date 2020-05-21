using PastaQ

N = 3
nshots = 50

# Method 1: direct gate application
psi = initializequbits(N)

applygate!(psi,"H",1)
applygate!(psi,"Cx",[1,2])
applygate!(psi,"Cx",[2,3])

samples = measure(psi,nshots)
@show samples
print("\n\n")

# Method 2: build circuit from gate_list
gate_list = [(gate = "H", site = 1),
             (gate = "Cx",site = [1,2]),
             (gate = "Cx",site = [2,3])]

circuit = makecircuit(psi,gate_list)

runcircuit!(psi,circuit)

samples = measure(psi,nshots)
@show samples
