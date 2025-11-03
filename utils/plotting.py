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

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	# qc = qiskit_circuit()
	# fig = qc.draw(output='mpl')
	# Display the figure
	# plt.show()
	# fig.savefig("qiskit_circuit.png")

	fig, ax = qml.draw_mpl(qml_circuit, decimals=2, style='default')()
	# plt.show()
	fig.savefig("qml_circuit.png")
	# print("\nPennyLane circuit:")
	# print(qml.draw(qml_circuit)() )
