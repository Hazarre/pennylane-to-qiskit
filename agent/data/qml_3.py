"""
Example 3: Noise Model Workflow - PennyLane Version
Define noise model targeting all 2-qubit gates
"""
import pennylane as qml
from pennylane import numpy as np

prob_depol = 0.05

# 1. Condition: "is this operation a 2-qubit gate?"
@qml.BooleanFn
def is_two_qubit_gate(op, **metadata):
    """Return True if the operation 'op' acts on exactly 2 wires."""
    return len(op.wires) == 2

# 2. Noise: apply single-qubit DepolarizingChannel on *each* wire of that gate
def two_qubit_depol(op, **metadata):
    """For any 2-qubit gate 'op', apply a depolarizing channel on each of its wires."""
    for w in op.wires:
        qml.DepolarizingChannel(prob_depol, wires=w)

# 3. Build the noise model: mapping from BooleanFn -> noise function
noise_model = qml.NoiseModel({is_two_qubit_gate: two_qubit_depol})

# 4. Base mixed-state device (supports channels)
base_dev = qml.device("default.mixed", wires=2, shots=1000)

# 5. Wrap device with noise model (device-level add_noise is recommended for channels)
noisy_dev = qml.add_noise(base_dev, noise_model)

# 6. Example circuit with a 2-qubit gate
@qml.qnode(noisy_dev)
def noisy_cnot_counts():
    """A simple circuit where CNOT triggers depolarizing noise on both qubits."""
    qml.Hadamard(0)        # unaffected by this particular noise model
    qml.CNOT(wires=[0, 1]) # <- noise will be inserted after this
    return qml.counts()

print("[PennyLane] Noisy counts:", noisy_cnot_counts())