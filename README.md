# pennylane-to-qiskit

## Work already done

- [Pennylane to Qiskit Guide](https://pennylane.ai/qml/demos/tutorial_guide_to_pennylane_knowing_qiskit)

* `from_qiskit()`: converts an entire Qiskit QuantumCircuit — the whole thing — into PennyLane. It will faithfully convert Qiskit-side measurements (even mid-circuit measurements) or you can append PennyLane-side measurements directly to it.
* `from_qiskit_op()`: converts a SparsePauliOp in Qiskit 1.0 to the equivalent operator in PennyLane.

# Context to provide

- Pennylane latest and runtime verion
- Qiskit latest and runtime verion
- Qiskit syntax is declarative and Pennylane is functional

# Mapping

- Gates:
  - Qml:
  - Qiskit: https://github.com/Qiskit/qiskit/tree/stable/2.2/qiskit/circuit/library/standard_gates
- Curcuit:
  -
- Measurement:
  - Qml: https://docs.pennylane.ai/en/stable/introduction/measurements.html
  - Qiskit: https://quantum.cloud.ibm.com/docs/en/guides/measure-qubits#mid-circuit-measurements
    QuantumCircuit.measure(qbit, cbit_to_store_results)
- Backend:

Qubits in PennyLane are called wires.
