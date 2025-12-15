"""
Example 2: Simulator Backend - PennyLane Version
Shot-based simulation with GHZ-like circuit
"""
import pennylane as qml

dev = qml.device("default.qubit", wires=3, shots=2000)

@qml.qnode(dev)
def ghz_counts():
    """Create a 3-qubit GHZ-like state and return shot counts."""
    qml.Hadamard(0)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    return qml.counts(wires=[0, 1, 2])

print("[PennyLane] counts:", ghz_counts())