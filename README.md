# Q-Tsuru

Quantum-Tsuru is a library for simulating quantum circuits and algorithms using PyTorch. Quantum states are represented as vectors (tensors), and all operations are applied directly on these vectors, ensuring efficiency for multi-qubit systems.

---

## Library Structure

### Core

| Module / Class      | Main purpose                                                                                   |
| ------------------- | ---------------------------------------------------------------------------------------------- |
| `BufferManager`     | Manages reusable tensor buffers, avoiding excessive memory allocation.                         |
| `BitOps`            | Efficient bitwise operations (set, clear, flip, permute, count bits).                          |
| `Operators`         | Applies quantum gates to state vectors using `BitOps` and `BufferManager`.                     |
| `States`            | Generates quantum state vectors in different bases and with fixed Hamming weight.             |
| `DiagonalOps`       | Constructs diagonal operators and Pauli-Z or number operator chains.                           |

---

### Hamiltonians

Hamiltonians define the evolution and cost of quantum states. All inherit from the abstract `Hamiltonian` class.

| Type        | Class / Example | Description                                                                                     |
| ----------- | ---------------- | ----------------------------------------------------------------------------------------------- |
| Abstract    | `Hamiltonian`    | Defines the interface for Hamiltonians, with methods `hamiltonian(psi)` and `evolution(psi, t)`.|
| Drivers     | `Hx`             | X driver: $H_x = \sum_i X_i$, applied to each qubit with Pauli-X.                               |
|             | `Hy`             | Y driver: $H_y = \sum_i Y_i$, applied to each qubit with Pauli-Y.                               |
|             | `Hz`             | Z driver: $H_z = \sum_i Z_i$, applied to each qubit with Pauli-Z.                               |
| Problems    | `Htsp`           | [TSP](https://doi.org/10.3389/fphy.2014.00005): encodes penalty constraints and the total route cost; allows fixing the first city.       |
|             | `Hmaxcut`        | [MaxCut](https://doi.org/10.3389/fphy.2014.00005): encode the Hamiltonian for the MaxCut problem. |
|             | `Hmst`        | [MST](https://doi.org/10.13140/RG.2.2.31829.73445): encode the Hamiltonian for the MST problem. |

---

### FQAs

Feedback-based Quantum Algorithms.

| Function     | Short description                |
| ------------ | -------------------------------- |
| `falqon`     | Feedback-based Algorithm for Quantum Optimization [(FALQON)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.250502)|
| `mdfalqon`     |FALQON for multiple driver Hamiltonians [(MD-FALQON)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.106.062414)|
| `sofalqon`   | Second Order FALQON [(SO-FALQON)](https://doi.org/10.1103/PhysRevResearch.7.013035)|
| `trfalqon`   | Time-Rescaled FALQON [(TR-FALQON)](https://doi.org/10.48550/arXiv.2504.01256)|









