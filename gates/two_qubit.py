from qiskit import QuantumCircuit
def qiskit_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc


import pennylane as qml
dev = qml.device("default.qubit", wires=2)
def qml_circuit():
	qml.Hadamard(wires=0)
	qml.CNOT(wires=[0, 1])
	return qml.state()

