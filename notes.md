# Pennylane

A quantum circuit in pennylane is designed to behave like a differentiable function.

## Discussion Nov 22nd

# QML program to Qisikit program via LLM agents

# Step 1

# Manual Contruction of a knowledge graph?

# How do we obtain translation edges? Ask LLM by providing examples of the format of the KG that I want along with the examples of matching QML and Qiskit programs.

```json
{
	id: 1,
	name: ,
	type: funtion | class ,
	description: ,
}

{
	from: node1
	to:
	relationship:
	description:
}
```

Pre-processing
Input: The docs, articles, or examples  
Output: Knowledge graph (Component nodes, Transformation edges)

Real time processing
Input: qml program -> AST -> Symbol Table -> Check if translation edge exists -> if it does, we feed that context to LLM.  
Output: qiskit program

# Step 2

# Write a prompt for the LLM to specialize in converting QML to Qiskit programs

# What context are we proving to the LLM?

- information about QML component (parser on sphinx documentation)
- information about Qiskit component (parser on sphinx documentation)
- relation bewteen the QML an Qiskit component (Step 1)

# Evaluation the output of the agent

- Agent that understands and compares the output of qml and Qiskit programs.
- Tooling callings ourselves and feed the result to LLM.
- Tool calling agent.

# Discussion Nov 2nd

- Did QASM already solve the conversion problem? If not, what is it missing?
- There's not always a mapping from Pennylane device to IBM device as the backend. Can it be assumed that either the user wants a simulator or IBM backend? For example, converting a Pennylane code that uses cirq backend would be somewhat out of context.

- Circuits are generally broken down into various components

  - Gates or operators: clear 1:1 mapping
  - Measurement
    - Qiskit: `QuantumCircuit.measure`, `measure_all`, `measure_active` and `MidCircuitMeasure`.
    - qml: `expval(op)`, `sample([op, wires, dtype])` and `counts([op, wires, all_outcomes])` seem the most relevant.
  - Backend, Device or Simulator:
    - default.qubit and lightning.qubit
    - StatevectorSampler and StatevectorEstimator

- Plotting

- What are some ways to verify conversion?
