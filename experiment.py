from phi.agent import Agent
from phi.model.groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

protobuf_agent = Agent(
    name="Protobuf Generator Agent",
    role="""You are an expert in creating Protocol Buffer (.proto) files. 
    You will receive a description of data structures and your task is to generate a valid .proto file.
    Ensure the generated .proto file is syntactically correct and accurately represents the provided data description.
    Include appropriate data types (string, int32, int64, float, double, bool), messages, and fields.
    If the provided description is ambiguous, make reasonable assumptions about the data structure.
    If the user asks for a modification, modify the existing proto file, do not create a new one from scratch unless asked to.
    """,
    model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    markdown=False,
)

def generate_and_save_proto(description, filename="output.proto"):
    """Generates a .proto file and saves it to the specified filename."""
    response = protobuf_agent.run(description)

    if response:
        try:
            with open(filename, "w") as f:
                f.write(response.content)
            print(f"Protobuf file saved to {filename}")
        except Exception as e:
            print(f"Error saving protobuf file: {e}")
    else:
        print("No protobuf content generated.")

# Example usage:
data_description = """Create a .proto file for a 'Product' with fields:
- name (string)
- id (int32)
- price (float)
- is_available (bool)
"""

generate_and_save_proto(data_description, filename="product.proto")

modification_description = """Modify the existing proto file to add a field 'description' (string) to the Product message."""

generate_and_save_proto(modification_description, filename="product.proto") #save to the same file to modify it.