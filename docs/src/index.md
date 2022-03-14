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

## FAQs
Zygote bug


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

## Research papers using PastaQ
If you used PastaQ.jl and your paper does not appear in this list, please let us know at [info@pastaq.org](mailto:info@pastaq.org).

**2022**   
- [2203.04948](https://arxiv.org/abs/2203.04948) *Fragile boundaries of tailored surface codes*, O Higgott, TC Bohdanowicz, A Kubica, ST Flammia, ET Campbell.     
**2021**   
- [2106.12627](https://arxiv.org/abs/2106.12627) *Provably efficient machine learning for quantum many-body problems*, H-Y Huang, R Kueng, G Torlai, VV Albert, J Preskill.     
- [2106.03769](https://arxiv.org/abs/2106.03769) *Measurement-induced phase transition in trapped-ion circuits*, S Czischek, G Torlai, S Ray, R Islam, RG Melko, *Phys. Rev. A 104, 062405*.       
- [2101.11099](https://arxiv.org/abs/2101.11099) *How To Use Neural Networks To Investigate Quantum Many-Body Physics*, J Carrasquilla and G Torlai, *PRX Quantum, 2, 040201*.

**2020**    
- [2009.01760](https://arxiv.org/abs/2009.01760) *Classical variational simulation of the Quantum Approximation Optimization Algorithm*, M Medvidovic and G Carleo, *Nature Communication, 7, 101*.    
- [2006.02424](https://arxiv.org/abs/2006.02424) *Quantum process tomography with unsupervised learning and tensor networks*, G Torlai, CJ Wood, A Acharya, G Carleo, J Carrasquilla, L Aolita.   

