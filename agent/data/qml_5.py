"""
Example 5: Variational Circuit + Optimization - PennyLane Version
Variational quantum circuit with gradient-based optimization
"""
import pennylane as qml
from pennylane import numpy as pnp

# PennyLane device: statevector-based simulator
dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev, interface="autograd")
def circuit(params):
    """PennyLane variational circuit:
    - Apply RY(theta) on qubit 0
    - Apply CNOT(0 -> 1)
    - Return expectation value of Z on qubit 1
    """
    qml.RY(params[0], wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(1))

# PennyLane built-in gradient-descent optimizer (uses analytic gradients where possible)
opt = qml.GradientDescentOptimizer(stepsize=0.1)

# Use PennyLane's numpy to ensure "trainable" parameters
params = pnp.array([0.5], requires_grad=True)

for i in range(20):
    params = opt.step(circuit, params)
    if i % 5 == 0:
        print(f"[PennyLane] Step {i}, cost = {circuit(params):.6f}")