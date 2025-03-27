from doc_parser import DocumentParser
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, Document
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",google_api_key=os.environ.get("GOOGLE_API_KEY"))

def invoke_llm(markdown_pages):
    if not isinstance(markdown_pages, list) or not all(isinstance(page, str) for page in markdown_pages):
        raise ValueError("Expected a list of Markdown strings.")

    markdown_content = "\n\n".join(markdown_pages)  

    prompt = """You are a Loss Analysis Expert. Your task is to extract structured data in JSON format from a Markdown-based loss report.

### **Extraction Rules:**
- Some claims span multiple lines.
- If a row has empty fields but contains `Detail Cause`, it belongs to the claim above.
- **Important distinctions:**
  - `injury_description`: First line describing the injury.
  - `detail_cause`: Second line (if present) explaining how the injury occurred.
  - If multiple `detail_cause` lines exist, **concatenate** them into a single string.
- **Do not merge `injury_description` and `detail_cause`.**  
- Preserve currency values **exactly** as they appear.
- Format output as:

```json
{
    "policy_number": "<policy_number>",
    "account_number": "<account_number>",
    "policy_name": "<policy_name>",
    "agency_name": "<agency_name>",
    "claim_details": [ 
        { 
            "claim_number": "<claim_number>",
            "claimant_name": "<claimant_name>",
            "carrier": "<carrier>",
            "claim_type": "<claim_type>",
            "status": "<status>",
            "injury_date": "",
            "report_date": "",
            "days_to_report": "",
            "closed_date": "",
            "injury_description": "<First injury description>",
            "body_part": "",
            "diagnosis": "",
            "detail_cause": "<Concatenated cause details>",
            "examiner": "",
            "department": "",
            "coverage": {
                "total_payments": "",
                "total_case_reserves": "",
                "total_incurred": "",
                "details": [ 
                    { 
                        "type": "<indemnity, medical, etc>",
                        "payment": "",
                        "case_reserve": "",
                        "incurred": "" 
                    } 
                ] 
            } 
        } 
    ] 
}

Loss Report (Markdown format follows):
"""


    full_input = prompt + "\n\n" + markdown_content
    document = Document(page_content=prompt + full_input)
    messages = [HumanMessage(content=document.page_content)]

    response = llm.invoke(messages)
    print(response.content)

custom_prompt = """
Strictly preserve markdown formatting during text extraction from scanned document.
"""

parser = DocumentParser(
    model_name="gemini-1.5-flash",
    api_key=os.environ.get("GOOGLE_API_KEY"),
    temperature=0.7,
    top_p=0.4,
    image_mode="url",
    detailed_extraction=False,
    enable_concurrency=True,
)

pdf_path = "Loss_Report.pdf" 
markdown_pages = parser.convert_pdf(pdf_path)
invoke_llm(markdown_pages)

# for i, page_content in enumerate(markdown_pages):
#     print(f"\n--- Page {i+1} ---\n{page_content}")
