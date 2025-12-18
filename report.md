# Pennylane to Qiskit Code Converter

## Overview

Build a large language model (LLM) agent that can utilize the knowledge graphs (KG) to help programmers convert Pennylane, a hardware agnostic quantum library focused on differentiable programming, to Qiskit, so they can take advantage of both libraries.

## Approach

At the highest level, the project context-engineers LLM agents such that they are knowledgeable about Qiskit, Pennlyane and how to translate between them. The context is provided to LLMs in the form of a knowledge graph, which consists of nodes that contain information about either Pennlylane or Qiskit interfaces, as well as edges that describe how to convert certain Pennylane code to Qiskit code. An example of a knowledge graph is:

````json
{
  "nodes": [
    {
      "id": "qml.RX",
      "name": "qml.RX",
      "type": "class",
      "description": "The single qubit X rotation gate in PennyLane. Applies a rotation around the X-axis of the Bloch sphere with matrix R_x(φ) = e^(-iφσ_x/2). Takes a rotation angle φ parameter and acts on a single wire."
    },
    {
      "id": "qiskit.circuit.QuantumCircuit.rx",
      "name": "qiskit.circuit.QuantumCircuit.rx",
      "type": "function",
      "description": "Method on QuantumCircuit class that adds an RX gate to the circuit. Applies a rotation around the X-axis of the Bloch sphere. Takes rotation angle and target qubit as parameters."
    }
  ],
  "edges": [
    {
      "from_": "qml.RX",
      "to_": "qiskit.circuit.QuantumCircuit.rx",
      "relationship": "convert to",
      "description": {
        "latest": "The PennyLane RX gate applies a rotation around the X-axis of the Bloch sphere. In Qiskit, this translates to the `rx()` method on a QuantumCircuit, which adds an RX gate to the circuit.\n\n**PennyLane Usage**\n```python\nimport pennylane as qml\n\ntape = qml.tape.QuantumTape()\nwith tape:\n    qml.RX(theta, wires=0)\n    qml.measure(0)\n```\n\n**Qiskit Equivalent**\n```python\nfrom qiskit import QuantumCircuit\n\nqc = QuantumCircuit(1)\nqc.rx(theta, 0)\nqc.measure_all()\n```\n\nBoth perform the same X-axis rotation operation on a qubit, with the angle parameter `theta` specifying the rotation amount in radians."
      }
    }
  ]
}
````

The LLM agent is then prompt-engineered with a system prompt that guides it to produce the output that we want. The quality of the responses from the agent can vary, so in the agentic loop, we execute the source Pennylane and target Qiskit programs and feed their outputs back to the LLM for interpretation on whether the two programs are producing semantically the same output or in other words, achieving the same task.

### Pre-processing – Construction of the knowledge graph

A combination of manual construction, document parsing, and LLM generation are used to construct the knowledge graph, which can then be indexed in a graph database for efficient LLM retrieval on only the context needed for a particular Pennylane-to-Qiskit conversion.

### Runtime – Agent loop

Before being fed to the agent, the source Pennylane program is parsed into its Abstract Syntax Tree (AST) to provide its structural context to the underlying LLM. Each node in the AST will be queried against the graph database to fetch parts of the knowledge graph that is relevant to the conversion of the source program.

All the provided context will be structured together with a system prompt and fed to an LLM that will return a target Qiskit program. Both the source Pennylane and target Qiskit programs will be executed, and their outputs will be either sent to an LLM or a human for interpretation of equivalence. If equivalence fails, the agent iterates and retries previous steps for a few times.

## Experiments

Currently our agent is at Stage 1, and we plan to experiment the 2nd and 3rd stages.

| Stage   | Knowledge Graph                     | Graph Database | LLM                                  | Outcomes                                                                            |
| ------- | ----------------------------------- | -------------- | ------------------------------------ | ----------------------------------------------------------------------------------- |
| Stage 1 | Manually constructed, small         | No             | Local model (Ollama 3)               | Target programs don't always run, may have sytax error, but at least look in shape. |
| Stage 2 | LLM generation, medium              | Maybe?         | Commercial API (Claude, Gemini, etc) | Not Ready                                                                           |
| Stage 3 | Parsed from docs and LLM generation | Yes            | Commercial API (Claude, Gemini, etc) | Not Ready                                                                           |

## Challenges, Learnings and Insights

The fundamental challenge for this project is establishing equivalence between a Pennylange and a Qiskit program, which are inherently different programming paradigms and designed for different purposes. To validate equivalence, whether a human or machine judge, must understand what the programs do and what their outputs mean. Knowledge about Pennylane, Qiskit and quantum programming that allows one to read program _between the lines_ is crucial for this project, but also takes time and practice to acquire. There are also limited non-structured examples on how to convert between Pennylane and Qiskit, meaning there's little available data to construct relation edges.

## Next Steps

### Use foundational models as opposed to small open-source models

Latest foundational models such as GPT-5, Claude Sonnet 4.5 and Gemini 2.5 consume computational resources beyond that of a personal computer, and can offer the best ability for coding tasks to date. Consuming APIs of these foundational models for our agent likely means more performant coding assistants than using small models like Ollama, which can run locally on a personal computer.

### Scale KG construction

Pennylange and Qiskit documentations are both built using Sphinx. Using the structure of Sphinx documentation, we can parse out interface descriptions to serve as informational nodes in KGs. Similarly, we find Pennylane and Qiskit programs that are of similar functions or existing examples of program conversion between the two as source data. With some formatting, these source data can serve as relation edges in an KG.

### Index KGs with database for retrieval

LLMs have a limited context window, and it’s best to only feed the most relevant context to them such that they don't side track or hallucinate. For this reason, we store and index the KG into a database for efficient retrieval. With that, only the informational nodes and relation edges relevant to the source program will be queried and provided as context to the LLMs in the agent loop.

### Human in the loop evaluation

Currently, we execute the source and target programs and instruct LLMs to interpret whether their outputs are producing the same results. This can be useful but isn’t always correct. We allow the option for the human user to provide feedback on where the target program can be enhanced, such that each iteration improves in the most meaningful way.

## Code

The source code for this project can be found at:
https://github.com/Hazarre/pennylane-to-qiskit.

## References

- [Practical GraphRAG: Making LLMs smarter with Knowledge Graphs](https://www.youtube.com/watch?v=XNneh6-eyPg)
- [Knowledge Graph Query Engine](https://developers.llamaindex.ai/python/examples/query_engine/knowledge_graph_query_engine/)
- [Tool calling with LangChain](https://blog.langchain.com/tool-calling-with-langchain/)
- https://datastax.github.io/graph-rag/examples/code-generation/
