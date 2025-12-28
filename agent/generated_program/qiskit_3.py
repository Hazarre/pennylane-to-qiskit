# Qiskit Conversion - Attempt 1

import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error
import numpy as np

# 1. Define noise parameters
prob_depol = 0.05

# 2. Build the noise model
# In Qiskit, a NoiseModel is built by adding specific QuantumError objects to gates.
noise_model = NoiseModel()

# Create a single-qubit depolarizing error.
# The `depolarizing_error(probability, num_qubits)` function creates a quantum error
# where `probability` is the total probability of an error occurring,
# and if it occurs, a random Pauli error is applied.
# For a single qubit, this corresponds directly to PennyLane's qml.DepolarizingChannel(p).
depol_error_1q = depolarizing_error(prob_depol, 1)

# Add this 1-qubit depolarizing error to all CNOT gates.
# Crucially, when a 1-qubit error is added to a multi-qubit gate (like 'cx'),
# Qiskit Aer's NoiseModel applies this error independently to each qubit of the gate
# after the gate execution. This directly mimics the PennyLane behavior where
# `two_qubit_depol` iterates through `op.wires` and applies a `DepolarizingChannel`
# to *each* wire independently after the 2-qubit gate.
# Gates like 'h' (Hadamard) are 1-qubit gates and are not targeted by this noise model.
noise_model.add_quantum_error(depol_error_1q, ['cx'])

# 3. Create a Qiskit Aer simulator with the noise model.
# The 'density_matrix' method is required for simulating with quantum channels (mixed states).
simulator = AerSimulator(method='density_matrix', noise_model=noise_model)

# 4. Define the example circuit (Qiskit's QuantumCircuit equivalent of a QNode)
# We need 2 qubits and 2 classical bits for measurement.
qc = QuantumCircuit(2, 2)

# Hadamard gate on qubit 0. This is a 1-qubit gate and is not targeted by the noise model.
qc.h(0)

# CNOT gate on qubits 0 and 1. This is a 2-qubit gate, so the depolarizing noise
# will be applied to qubit 0 and qubit 1 independently after this gate.
qc.cx(0, 1)

# Measure all qubits and map them to classical bits.
qc.measure([0, 1], [0, 1])

# 5. Execute the circuit on the noisy simulator.
shots = 1000
# Adding a seed for reproducibility, similar to how PennyLane's default.mixed simulator
# might produce consistent results for repeated runs without explicit shot seeding.
job = simulator.run(qc, shots=shots, seed_simulator=123)
result = job.result()

# 6. Get measurement counts from the simulation result.
noisy_counts_qiskit = result.get_counts(qc)

print("[Qiskit] Noisy counts:", noisy_counts_qiskit)