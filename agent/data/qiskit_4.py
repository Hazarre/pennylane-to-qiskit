"""
Example 4: IBM Quantum Hardware - Qiskit Version
Running on real IBM Quantum hardware using EstimatorV2
"""
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.quantum_info import SparsePauliOp

try:
    # Connect to IBM Quantum Runtime
    service = QiskitRuntimeService()
    backend = service.backend("ibm_sherbrooke")

    # Logical circuit: note there are NO measurements here, since EstimatorV2
    # expects circuits without classical measurements.
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    # Observable: Z on qubit 1 (same convention as before)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])

    # Compile logical circuit to backend's native gate set & layout
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    isa_circuit = pm.run(qc)

    # Update observable according to new layout (physical qubits vs logical)
    isa_observable = obs.apply_layout(isa_circuit.layout)

    # Runtime EstimatorV2 in 'mode=backend' executes on the target backend
    estimator = EstimatorV2(mode=backend)
    job = estimator.run([(isa_circuit, isa_observable)])
    result = job.result()[0]

    print("[Qiskit]    Hardware expval(Z_1):", result.data.evs)

except Exception as e:
    print("[Qiskit]    Skipping hardware example due to:", e)