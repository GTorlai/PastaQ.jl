<img src="assets/pastaq-logo.jpg" width="600"/>

[![Tests](https://github.com/GTorlai/PastaQ.jl/workflows/Tests/badge.svg)](https://github.com/GTorlai/PastaQ.jl/actions?query=workflow%3ATests)
[![codecov](https://codecov.io/gh/GTorlai/PastaQ.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/GTorlai/PastaQ.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://gtorlai.github.io/PastaQ.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://gtorlai.github.io/PastaQ.jl/dev/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![website](https://img.shields.io/badge/website-pastaq.org-orange.svg)](https://www.pastaq.org/)
[![arXiv](https://img.shields.io/badge/arXiv--b31b1b.svg)](https://arxiv.org/abs/)

PLEASE NOTE THIS IS PRE-RELEASE SOFTWARE

# PastaQ.jl: design and benchmarking quantum hardware
PastaQ.jl is a Julia software toolbox providing a range of computational methods for quantum computing applications. These include the simulation of quancum circuits, the design of quantum gates, noise characterization and performance benchmarking, among others. PastaQ relies on tensor-network representations of quantum states and processes, and borrows well-refined techniques from the field of machine learning and data science, such as probabilistic modeling and automatic differentiation.

![alt text](assets/readme_summary.jpg)

---
## Install
The PastaQ package can be installed with the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run:

```julia
julia> ]

pkg> add PastaQ
```

PastaQ.jl relies on the following packages: [ITensors.jl](https://github.com/ITensor/ITensors.jl) for low-level tensor-network algorithms, [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) for stochastic optimization methods, [Zygote.jl](https://github.com/FluxML/Zygote.jl) for automatic differentiation, and [Observers.jl](https://github.com/GTorlai/Observers.jl) for tracking/recording metrics and observables. Please note that right now, PastaQ.jl requires that you use Julia v1.6 or later. 

---


## Documentation

- [**STABLE**](https://gtorlai.github.io/PastaQ.jl/stable/) --  **documentation of the most recently tagged version.**
- [**DEVEL**](https://gtorlai.github.io/PastaQ.jl/dev/) -- *documentation of the in-development version.*

## Examples
We briefly showcase some of the functionalities provided by PastaQ.jl. For more in-depth discussion, please refer to the tutorials folder.

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
