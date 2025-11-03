from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

def qiskit_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc
qc = qiskit_circuit()
qc.measure_all()

sampler = StatevectorSampler()
job_sampler = sampler.run([qc], shots=1024)
result_sampler = job_sampler.result()[0].data.meas.get_counts()
print(result_sampler)



import pennylane as qml
dev = qml.device("default.qubit")
@qml.set_shots(1024)
@qml.qnode(dev)
def pl_circuit():
    """
    Equivalent to doing:
    pl_circuit = qml.QNode(qml.from_qiskit(qc, measurements=qml.counts(wires=[0, 1])), dev)
    """
    qml.Hadamard(0)
    qml.CNOT([0, 1])
    return qml.counts(wires=[0, 1])
pl_circuit = qml.set_shots(qml.QNode(pl_func, dev), shots = 1024)
print(pl_circuit())