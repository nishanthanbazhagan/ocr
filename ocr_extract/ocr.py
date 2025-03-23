from llama_extract import LlamaExtract
import os
import json

# bring in our LLAMA_CLOUD_API_KEY
from dotenv import load_dotenv
load_dotenv()

# Initialize client
extractor = LlamaExtract(api_key=os.environ.get("LLAMA_CLOUD_API_KEY"))
print(extractor)

# List all agents

agent = extractor.get_agent(name="OCR_2" )
print(agent.data_schema )

oc = agent.extract("Loss_Report.pdf" )
print(oc.data)
output_file = "extracted_loss_report.json"
try:
    parsed_json = oc.data # Ensure valid JSON format
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parsed_json, f, indent=4)
    print(f"Extracted JSON saved to {output_file}")
except json.JSONDecodeError:
    print("Error: LLM response is not valid JSON.")
    with open("error_output.txt", "w", encoding="utf-8") as f:
        f.write(oc.data)  # Save raw output for debugging
    print("Raw response saved to error_output.txt")
