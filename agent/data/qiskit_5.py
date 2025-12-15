"""
Example 5: Variational Circuit + Optimization - Qiskit Version
Variational quantum circuit with gradient-based optimization using EstimatorV2
"""
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2 as AerEstimator  # EstimatorV2 implementation (simulator)
from qiskit_algorithms.optimizers import GradientDescent

import numpy as np

# Define a parameterized 2-qubit circuit
qc = QuantumCircuit(2)
theta = Parameter('theta')
qc.ry(theta, 0)
qc.cx(0, 1)

# Observable: Z on qubit 1.
# In Qiskit Pauli strings, the RIGHT-most character is qubit 0.
# For 2 qubits, "ZI" = Z on qubit 1, I on qubit 0.
obs = SparsePauliOp.from_list([("ZI", 1.0)])

# EstimatorV2 from Aer: takes (circuit, observable, parameter_values)
est = AerEstimator()

def cost(x: np.ndarray) -> float:
    """Compute expectation value <Z_1> for given parameters x = [theta]."""
    t = float(x[0])
    # EstimatorV2.run takes a list of "pubs": (circuit, observables, parameter_values)
    pubs = [(qc, obs, [[t]])]

    job = est.run(pubs)
    pub_result = job.result()[0]        # first (and only) pub
    ev = np.asarray(pub_result.data.evs).item()  # scalar exp. value

    return float(ev)

# Set up GradientDescent optimizer (finite-difference gradient by default)
initial_point = np.array([0.5], dtype=float)

iter_counter = {"i": 0}
def callback(nfev, x, fx, grad_norm):
    """Log every 5th optimization step."""
    i = iter_counter["i"]
    if i % 5 == 0:
        print(f"[Qiskit]    Step {i}, cost = {fx:.6f}")
    iter_counter["i"] += 1

opt = GradientDescent(
    maxiter=20,
    learning_rate=0.1,
    callback=callback,
)

result = opt.minimize(fun=cost, x0=initial_point)