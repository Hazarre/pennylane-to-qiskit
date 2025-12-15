"""
Example 1: Gate Translation - PennyLane Version
Single-qubit gate (qml.RX) circuit construction
"""
import pennylane as qml

theta = 0.3

# Use a QuantumTape to *record* operations, not execute them.
tape = qml.tape.QuantumTape()
with tape:
    qml.RX(theta, wires=0)
    qml.measure(0)

# Print the PennyLane circuit structure
print(tape.draw())