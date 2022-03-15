# Quantum States


## Hilbert spaces 

```@docs
qubits
qudits
```
## Product states

```@docs
productstate
```

By default, the product states that can be built ouf ot the box are based
on the single-qubit Pauli eigenstates ``\{|0\rangle,|1\rangle,|+\rangle,|-\rangle,|i\rangle,|-i\rangle\}``.
A different set of elementary quantum states can be easily defined as follows:
```julia
import ITensors: state
state(::StateName"mystate", ::SiteType"Qubit") = [1/√3, √2/√3]
```

## Random states



