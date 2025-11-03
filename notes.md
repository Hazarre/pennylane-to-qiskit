# Pennylane

A quantum circuit in pennylane is designed to behave like a differentiable function.

# Discussion Nov 2nd

- Did QASM already solve the conversion problem? If not, what is it missing?
- There's not always a mapping from Pennylane device to IBM device as the backend. Can it be assumed that either the user wants a simulator or IBM backend? For example, converting a Pennylane code that uses cirq backend would be somewhat out of context.

- Circuits are generally broken down into various components

  - Gates or operators: clear 1:1 mapping
  - Measurement
    - Qiskit: `QuantumCircuit.measure`, `measure_all`, `measure_active` and `MidCircuitMeasure`.
    - qml: `expval(op)`, `sample([op, wires, dtype])` and `counts([op, wires, all_outcomes])` seem the most relevant.
  - Backend, Device or Simulator:
    - default.qubit and lightning.qubit
    - StatevectorSampler and StatevectorEstimator

- Plotting

- What are some ways to verify conversion?
