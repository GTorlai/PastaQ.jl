# PastaQ: design and benchmarking quantum hardware
PastaQ.jl is a Julia software toolbox providing a range of computational 
methods for quantum computing applications. Some examples are the simulation of quancum circuits, the design of quantum gates, noise characterization and performance benchmarking. PastaQ relies on tensor-network representations of quantum states and processes, and borrows well-refined techniques from the field of machine learning and data science, such as probabilistic modeling and automatic differentiation.

---
## Install
The PastaQ package can be installed with the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run:

```julia
julia> ]

pkg> add PastaQ
```

PastaQ.jl relies on the following packages: [ITensors.jl](https://github.com/ITensor/ITensors.jl) for low-level tensor-network algorithms, [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) for stochastic optimization methods, [Zygote.jl](https://github.com/FluxML/Zygote.jl) for automatic differentiation, and [Observers.jl](https://github.com/GTorlai/Observers.jl) for tracking/recording metrics and observables. Please note that right now, PastaQ.jl requires that you use Julia v1.6 or later. 

---

## Getting started
In this simple example, we show how to build and run a quantum circuit to generate
the ``n``-qubit GHZ state
```math
|GHZ\rangle = \frac{1}{\sqrt{2}}(|0,0,\dots,0\rangle + |1,1,\dots,1\rangle).
```
The circuit consists of a single Hadamard gate on qubit `1`, followed by a set of 
controlled-NOT gates between pairs of qubits.

In PastaQ, a quantum gates is described by a data structure `g = ("gatename", support, params)`, 
consisting of a gate identifier gatename (`String`), a support (an `Int` for single-qubit gates 
or a `Tuple` for multi-qubit gates), and a set of gate parameters, such as rotations angles, whenever needed. 
A comprehensive set of elementary gates is provided, including Pauli operations, phase and T gates, 
single-qubit rotations, controlled gates, Toffoli gate and others. 
Additional user-specific gates can be easily added to this collection.

```julia
using PastaQ

# number of qubits
n = 20

# manually create a circuit to prepare GHZ state,
# or use built-in call `circuit = ghz(n)`
circuit = [("H", 1),]
for j in 1:n-1
  circuit = [circuit; ("CX", (j, j+1))]
end
```
```julia
20-element Vector{Tuple{String, Any}}:
 ("H", 1)
 ("CX", (1, 2))
 ("CX", (2, 3))
 ...
 ("CX", (18, 19))
 ("CX", (19, 20))
```

In order to execute a circuit, we first define the Hilbert space of 
our system, and then run the circuit using the `runcircuit` function.
While the first step is not strictly necessary (i.e. the Hilbert space
is generated internally if not provided, it is best practice to do so,
so that various objects can be defined on the same Hilbert space, a 
requirement to be able to perform calculations (such as inner products,
expectation values etc). The `runcircuit` function in this case generates
an output MPS wavefunction corresponding to the GHZ state. 

```julia
# run the circuit to obtain the output MPS
hilbert = qubits(n)
ψ = runcircuit(hilbert, circuit)
```
```julia
ITensors.MPS
[1] ((dim=2|id=187|"Qubit,Site,n=1"), (dim=2|id=980|"Link,n=1"))
[2] ((dim=2|id=845|"Qubit,Site,n=2"), (dim=2|id=980|"Link,n=1"), (dim=2|id=245|"Link,n=1"))
[3] ((dim=2|id=559|"Qubit,Site,n=3"), (dim=2|id=245|"Link,n=1"), (dim=2|id=175|"Link,n=1"))
...
[19] ((dim=2|id=903|"Qubit,Site,n=19"), (dim=2|id=639|"Link,n=1"), (dim=2|id=66|"Link,n=1"))
[20] ((dim=2|id=66|"Link,n=1"), (dim=2|id=212|"Qubit,Site,n=20"))
```

We can then perform a set of operations on the output MPS, such as generating
a set of projective measurements in the computational basis:
```julia
# sample projective measurements in the computational basis
getsamples(ψ, 5)
```
```julia
5×20 Matrix{Int64}:
1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1
```

PastaQ also supports circuit with noisy gates. A noise model is defined in a 
similar way to quantum gates, by a `String` identifier of the specific noise
model, and additional parameters (such as noise strength etc). Here we define
a depolarizing Kraus channel with different strengths for one- and two-qubit
gates. By running the circuit with the keyword argument `noise = noisemodel`, 
the corresponding Kraus operators are attached to each gate in the circuit. The
output is a MPO density operator:
```julia
# define a noise model with different error rates for
# one- and two-qubit gates
noisemodel = (1 => ("depolarizing", (p = 0.01,)), 
              2 => ("depolarizing", (p = 0.05,)))

# run a noisy circuit
ρ = runcircuit(hilbert, circuit; noise = noisemodel)
```
```julia
ITensors.MPO
[1] ((dim=2|id=187|"Qubit,Site,n=1")', (dim=2|id=187|"Qubit,Site,n=1"), (dim=4|id=104|"Link,n=1"))
[2] ((dim=2|id=845|"Qubit,Site,n=2")', (dim=4|id=104|"Link,n=1"), (dim=2|id=845|"Qubit,Site,n=2"), (dim=4|id=668|"Link,n=1"))
...
[19] ((dim=2|id=903|"Qubit,Site,n=19")', (dim=4|id=792|"Link,n=1"), (dim=2|id=903|"Qubit,Site,n=19"), (dim=4|id=982|"Link,n=1"))
[20] ((dim=4|id=982|"Link,n=1"), (dim=2|id=212|"Qubit,Site,n=20")', (dim=2|id=212|"Qubit,Site,n=20"))
```

---

## Citation
If you use PastaQ.jl in your work, for now please consider citing the Github page:

```
@misc{pastaq,
    title={\mbox{PastaQ}: A Package for Simulation, Tomography and Analysis of Quantum Computers},
    author={Giacomo Torlai and Matthew Fishman},
    year={2020},
    url={https://github.com/GTorlai/PastaQ.jl/}
}
```

--- 

## Research papers using PastaQ
If you used PastaQ.jl and your paper does not appear in this list, please let us know at [info@pastaq.org](mailto:info@pastaq.org).

**2022**   
- [2203.04948](https://arxiv.org/abs/2203.04948) *Fragile boundaries of tailored surface codes*, O Higgott, TC Bohdanowicz, A Kubica, ST Flammia, ET Campbell.     
**2021**   
- [2106.12627](https://arxiv.org/abs/2106.12627) *Provably efficient machine learning for quantum many-body problems*, H-Y Huang, R Kueng, G Torlai, VV Albert, J Preskill.     
- [2106.03769](https://arxiv.org/abs/2106.03769) *Measurement-induced phase transition in trapped-ion circuits*, S Czischek, G Torlai, S Ray, R Islam, RG Melko, *Phys. Rev. A 104, 062405*.       

**2020**    
- [2009.01760](https://arxiv.org/abs/2009.01760) *Classical variational simulation of the Quantum Approximation Optimization Algorithm*, M Medvidovic and G Carleo, *Nature Communication, 7, 101*.    
- [2006.02424](https://arxiv.org/abs/2006.02424) *Quantum process tomography with unsupervised learning and tensor networks*, G Torlai, CJ Wood, A Acharya, G Carleo, J Carrasquilla, L Aolita.   

