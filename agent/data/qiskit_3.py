"""
Example 3: Noise Model Workflow - Qiskit Version
Define noise model targeting all 2-qubit gates using Aer NoiseModel
"""
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, errors

prob_depol = 0.05

# -------------------------------
# 1. Build noise model
# -------------------------------

noise_model = NoiseModel()

# List of common 2-qubit gate names to target with noise.
# You can extend this list as needed for your backend.
two_qubit_gates = [
    "cx", "cz", "swap",
    "iswap", "ecr", "rxx", "ryy", "rzz", "rzx",
]

# 2-qubit depolarizing error (acts jointly on the 2-qubit subspace)
twoq_error = errors.depolarizing_error(prob_depol, 2)

# Attach this error to *all* 2-qubit gates on *all* qubits
noise_model.add_all_qubit_quantum_error(twoq_error, two_qubit_gates)

# -------------------------------
# 2. Example circuit
# -------------------------------

qc = QuantumCircuit(2)
qc.h(0)             # 1-qubit gate, NOT noised by our model
qc.cx(0, 1)         # 2-qubit gate, WILL have depolarizing error applied
qc.measure_all()

# -------------------------------
# 3. Simulate with noisy backend
# -------------------------------

sim = AerSimulator(noise_model=noise_model, shots=1000)
result = sim.run(qc).result()

print("[Qiskit]    Noisy counts:", result.get_counts())