"""
Example 1: Gate Translation - Qiskit Version
Single-qubit gate (RXGate) circuit construction
"""
from qiskit import QuantumCircuit

qc = QuantumCircuit(1)
theta = 0.3
qc.rx(theta, 0)
qc.measure_all()

# Print the Qiskit circuit structure
print(qc)