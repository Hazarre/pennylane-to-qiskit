"""
Example 4: IBM Quantum Hardware - PennyLane Version
Running on real IBM Quantum hardware using qiskit.remote device
"""
import pennylane as qml
from qiskit_ibm_runtime import QiskitRuntimeService

try:
    # Connect to IBM Quantum / IBM Cloud account
    service = QiskitRuntimeService()

    # Choose a backend (explicit name used here; you can also call least_busy())
    backend = service.backend("ibm_sherbrooke")

    # Use the Pennylane-Qiskit "qiskit.remote" device to talk to Runtime backends
    num_qubits_supported = backend.num_qubits
    dev = qml.device("qiskit.remote", wires=num_qubits_supported, backend=backend)

    @qml.qnode(dev)
    def pl_hardware_circuit():
        """Simple 2-qubit circuit evaluated on IBM hardware via PennyLane."""
        qml.Hadamard(0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(1))

    print("[PennyLane] Hardware expval(Z_1):", pl_hardware_circuit())

except Exception as e:
    print("[PennyLane] Skipping hardware example due to:", e)