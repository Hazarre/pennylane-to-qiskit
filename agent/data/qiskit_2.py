"""
Example 2: Simulator Backend - Qiskit Version
Shot-based simulation with GHZ-like circuit using AerSimulator
"""
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.measure_all()

sim = AerSimulator(shots=2000)
result = sim.run(qc).result()
counts = result.get_counts()

print("[Qiskit]    counts:", counts)