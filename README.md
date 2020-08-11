![alt text](https://github.com/GTorlai/PastaQ.jl/blob/master/docs/pastaQ_logo.jpg?raw=true)

PLEASE NOTE THIS IS PRE-RELEASE SOFTWARE 

EXPECT ROUGH EDGES AND BACKWARD INCOMPATIBLE UPDATES

# A Package for Simulation, Tomography and Analysis of Quantum Computers

PastaQ is a julia package for simulation and benchmarking of quantum computers using a combination of machine learning and tensor-network algorithms.

The main features of PastaQ are:
+ **Simulation of quantum circuits**. The package provides a simulator based on Matrix Product States (MPS) to simulate quantum circuits compiled into a set of quantum gates. Noisy circuits are simulated by specifying a noise model, which is applied to each quantum gate. We show as examples two quantum algorithms: the quantum Fourier transforrm and the variational quantum eigensolver.
+ **Quantum state tomography**. This module implements the reconstruction of an unknown quantum state from measurement data. Depending whether the quantum state is pure or mixed, the variational ansatz implemented is an MPS or a locally-purified density operators (LPDO). The measurement data consists of a set of measurement bases and the corresponding bit-strings of the measurement outcome. The reconstruction is realized by minimizing a statistical divergence between the data and the tensor-network probability distributions.
+ **Quantum process tomography**. This module implements the reconstruction of an unknown quantum channel from measurements.

PastaQ is developed at the Center for Computational Quantum Physics of the Flatiron Institute, and it is supported by the Simons Foundation.

