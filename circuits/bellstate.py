from qiskit import QuantumCircuit

def qiskit_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc
qc = qiskit_circuit()

import pennylane as qml
pl_func = qml.from_qiskit(qc)
