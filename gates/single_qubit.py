from qiskit import QuantumCircuit
def qiskit_circuit():
    qc = QuantumCircuit(1)
    qc.h(0)
    return qc

import pennylane as qml
dev = qml.device("default.qubit", wires=1)
@qml.qnode(dev)
def qml_circuit():
	qml.Hadamard(wires=0)
	return qml.state()

# if __name__ == "__main__":
# 	print("Qiskit circuit:")
# 	print(qiskit_circuit())
	
# 	print("\nPennyLane circuit:")
# 	print(qml.draw(qml_circuit)() )
