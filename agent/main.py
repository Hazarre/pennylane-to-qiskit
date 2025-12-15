from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import sys
import httpx
import json
import re
import subprocess
import tempfile
import os

def create_coding_assistant(system_prompt: str, model: str = "mistral:7b"):
	"""
	Create a coding assistant using Ollama and LangChain.
	
	Args:
		system_prompt: The system prompt to guide the assistant
		model: The Ollama model to use (default: mistral)
	
	Returns:
		A callable assistant function
	"""
	llm = ChatOllama(model=model, temperature=0.1)
	
	prompt_template = PromptTemplate(
		input_variables=["system_prompt", "user_input"],
		template="{system_prompt}\n\nUser: {user_input}\n\nAssistant:"
	)
	
	# Create a chain using the | operator (modern LangChain approach)
	chain = prompt_template | llm
	
	def assistant(user_input: str) -> str:
		"""Execute the assistant with user input."""
		try:
			response = chain.invoke({
				"system_prompt": system_prompt,
				"user_input": user_input
			})
			return response.content.strip()
		except httpx.ConnectError as e:
			return f"❌ Error: Cannot connect to Ollama server.\n\nPlease ensure Ollama is running:\n1. Install Ollama: https://ollama.ai/\n2. Start Ollama: 'ollama serve'\n3. Pull the model: 'ollama pull {model}'\n\nOriginal error: {str(e)}"
		except Exception as e:
			return f"❌ Unexpected error: {str(e)}"
	
	return assistant


def extract_qiskit_conversion(response: str) -> str:
	"""Extract the Qiskit conversion code from the LLM response."""
	# Look for code between <qiskit_conversion> tags
	match = re.search(r'<qiskit_conversion>(.*?)</qiskit_conversion>', response, re.DOTALL)
	if match:
		content = match.group(1).strip()
		# Remove any markdown code blocks within the XML tags
		content = re.sub(r'```python\s*\n?', '', content)
		content = re.sub(r'\n?\s*```\s*$', '', content)
		return content.strip()
	
	# Fallback: look for Python code blocks
	code_blocks = re.findall(r'```python\s*\n(.*?)\n\s*```', response, re.DOTALL)
	if code_blocks:
		return code_blocks[-1].strip()  # Take the last code block
	
	return ""


def execute_python_code(code: str) -> tuple[str, str]:
	"""Execute Python code and return stdout and stderr."""
	try:
		# Create a temporary file
		with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
			f.write(code)
			temp_file = f.name
		
		# Execute the code
		result = subprocess.run(
			[sys.executable, temp_file],
			capture_output=True,
			text=True,
			timeout=30
		)
		
		# Clean up
		os.unlink(temp_file)
		
		return result.stdout, result.stderr
		
	except subprocess.TimeoutExpired:
		if 'temp_file' in locals():
			os.unlink(temp_file)
		return "", "Code execution timed out after 30 seconds"
	except Exception as e:
		if 'temp_file' in locals():
			os.unlink(temp_file)
		return "", f"Error executing code: {str(e)}"


def main():
	# Load system prompt template
	sysprompt = "sysprompt_twostep.md" 
	knowledge_graph_file = "knowledge_graph/p2q_kg.json"
	input_program = "data/qml_1.py"  # Updated path to use the files we created earlier

	try:
		with open(sysprompt, "r") as f:
			system_prompt_template = f.read()
	except FileNotFoundError:
		system_prompt_template = "You are a helpful coding assistant that specializes in translating PennyLane quantum code to Qiskit.\n\nPennyLane program:\n{{PENNYLANE_PROGRAM}}"
	
	# Load knowledge graph
	try:
		with open(knowledge_graph_file, "r") as f:
			knowledge_graph = json.load(f)
			kg_str = json.dumps(knowledge_graph, indent=2)
	except FileNotFoundError:
		kg_str = "No knowledge graph available"
	
	# Create a function that substitutes templates and creates assistant
	def create_assistant_with_program(pennylane_program: str):
		# Substitute template variables in system prompt
		system_prompt = system_prompt_template.replace("{{KNOWLEDGE_GRAPH}}", kg_str)
		system_prompt = system_prompt.replace("{{PENNYLANE_PROGRAM}}", pennylane_program)
		return create_coding_assistant(system_prompt)
	
	# Test Ollama connection with minimal assistant
	print("Testing Ollama connection...")
	test_assistant = create_coding_assistant("You are a test assistant.")
	test_response = test_assistant("test")
	if "❌ Error" in test_response:
		print(test_response)
		return
	
	# Load PennyLane program from file instead of console input
	try:
		with open(input_program, "r") as f:
			qml_input = f.read()
	except FileNotFoundError:
		print(f"❌ Error: {input_program} not found.")
		return
	
	# Create assistant with the loaded PennyLane program
	assistant = create_assistant_with_program(qml_input)
	
	max_attempts = 3
	for attempt in range(1, max_attempts + 1):
		print(f"\n{'='*50}\nATTEMPT {attempt}/{max_attempts}\n{'='*50}")
		
		# STEP 1: Generate conversion (up to <qiskit_conversion> only)
		print("=== STEP 1: Generating Qiskit conversion ===")
		conversion_prompt = f"Please analyze the PennyLane program and generate the Qiskit conversion for attempt {attempt}. Stop after the <qiskit_conversion> section - do not proceed to validation yet."
		conversion_response = assistant(conversion_prompt)
		print(f"\nConversion Response:\n{conversion_response}\n")
		
		# Extract and execute the Qiskit code
		qiskit_code = extract_qiskit_conversion(conversion_response)
		if not qiskit_code:
			print("❌ Could not extract Qiskit code from response")
			continue
		
		print("=== STEP 2: Executing Qiskit code ===")
		print(f"Extracted code:\n{qiskit_code}\n")
		
		stdout, stderr = execute_python_code(qiskit_code)
		execution_context = f"\nEXECUTION RESULTS:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
		
		print(f"Execution output:{execution_context}\n")
		
		# STEP 3: Continue with validation using execution results
		print("=== STEP 3: Validation with execution results ===")
		validation_prompt = f"""Now continue with the validation step for attempt {attempt}. Here are the execution results from the Qiskit program:
		{execution_context}
		
		Please complete the <validation> and <evaluation> sections, and provide your <final_answer> if this attempt is successful or if this is the final attempt.
		
		Previous conversion response:
		{conversion_response}"""
		
		validation_response = assistant(validation_prompt)
		print(f"\nValidation Response:\n{validation_response}\n")
		
		# Check if the attempt was successful
		if "<success>YES</success>" in validation_response:
			print("✅ Conversion successful!")
			break
		elif attempt == max_attempts:
			print(f"❌ All {max_attempts} attempts completed without success.")
		else:
			print(f"⚠️ Attempt {attempt} failed, trying again...")


if __name__ == "__main__":
	main()