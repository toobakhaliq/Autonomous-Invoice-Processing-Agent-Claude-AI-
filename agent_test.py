from litellm import completion
import json
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY not found in environment")

# -----------------------------
# Tool Registry
# -----------------------------
_TOOL_REGISTRY = {}


def register_tool(tags: List[str]):
    """Decorator to register tools with tags."""

    def decorator(func):
        _TOOL_REGISTRY[func.__name__] = {
            "function": func,
            "tags": tags,
            "description": func.__doc__
        }
        return func

    return decorator


# -----------------------------
# LLM Helper Functions
# -----------------------------
def call_llm(prompt: str, system_prompt: Optional[str] = None) -> str:
    """Call LiteLLM with Claude."""
    messages = [{"role": "user", "content": prompt}]

    kwargs = {
        "model": "claude-sonnet-4-5-20250929",
        "messages": messages,
        "api_key": ANTHROPIC_API_KEY
    }

    if system_prompt:
        kwargs["system"] = system_prompt

    response = completion(**kwargs)
    return response.choices[0].message.content


def prompt_expert(description_of_expert: str, prompt: str) -> str:
    """
    Call an LLM with an expert persona.

    Args:
        description_of_expert: Description of the expert persona
        prompt: The task/question for the expert

    Returns:
        Expert's response
    """
    return call_llm(prompt=prompt, system_prompt=description_of_expert)


def prompt_llm_for_json(schema: dict, prompt: str) -> dict:
    """
    Call LLM and parse JSON response.

    Args:
        schema: Expected JSON schema
        prompt: The prompt

    Returns:
        Parsed JSON dictionary
    """
    response = call_llm(prompt)

    # Clean up response
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Return default based on schema
        return {prop: None for prop in schema.get("properties", {}).keys()}


# -----------------------------
# Categorization Expert
# -----------------------------
@register_tool(tags=["invoice_processing", "categorization"])
def categorize_expenditure(description: str) -> str:
    """
    Categorize an invoice expenditure based on a short description.

    Args:
        description: A one-sentence summary of the expenditure.

    Returns:
        A category name from the predefined set of 20 categories.
    """
    categories = [
        "Office Supplies", "IT Equipment", "Software Licenses", "Consulting Services",
        "Travel Expenses", "Marketing", "Training & Development", "Facilities Maintenance",
        "Utilities", "Legal Services", "Insurance", "Medical Services", "Payroll",
        "Research & Development", "Manufacturing Supplies", "Construction", "Logistics",
        "Customer Support", "Security Services", "Miscellaneous"
    ]

    prompt = f"""Given the following description: '{description}', classify the expense into ONE of these categories:
{', '.join(categories)}

Respond with ONLY the category name, nothing else."""

    return prompt_expert(
        description_of_expert="You are a senior financial analyst with deep expertise in corporate spending categorization.",
        prompt=prompt
    )


# -----------------------------
# Compliance Expert
# -----------------------------
@register_tool(tags=["invoice_processing", "validation"])
def check_purchasing_rules(invoice_data: dict) -> dict:
    """
    Validate an invoice against company purchasing policies.

    Args:
        invoice_data: Extracted invoice details, including vendor, amount, and line items.

    Returns:
        A dictionary indicating whether the invoice is compliant, with explanations.
    """
    # Load the latest purchasing rules from disk
    rules_path = "config/purchasing_rules.txt"

    try:
        with open(rules_path, "r") as f:
            purchasing_rules = f.read()
    except FileNotFoundError:
        purchasing_rules = "No specific rules available. General corporate purchasing guidelines apply."

    validation_schema = {
        "type": "object",
        "properties": {
            "compliant": {"type": "boolean"},
            "issues": {"type": "string"}
        }
    }

    prompt = f"""Given this invoice data: {json.dumps(invoice_data, indent=2)}, check whether it complies with company purchasing rules.

The latest purchasing rules are as follows:
{purchasing_rules}

Respond with a JSON object containing:
- "compliant": true if the invoice follows all policies, false otherwise
- "issues": A brief explanation of any violations or missing requirements (empty string if compliant)

Respond ONLY with valid JSON, no other text."""

    return prompt_llm_for_json(
        schema=validation_schema,
        prompt=prompt
    )


# -----------------------------
# Invoice Parser
# -----------------------------
def extract_invoice_data(text: str) -> Optional[dict]:
    """
    Extract structured data from invoice text.

    Args:
        text: Raw invoice text

    Returns:
        Dictionary with invoice data or None if parsing fails
    """
    try:
        invoice_match = re.search(r"Invoice\s+#(\d+)", text)
        vendor_match = re.search(r"Vendor:\s*(.*?)(?:\n|$)", text)
        total_match = re.search(r"Total:\s*\$(\d+(?:,\d{3})*)", text)

        if not all([invoice_match, vendor_match, total_match]):
            return None

        invoice_number = invoice_match.group(1)
        vendor = vendor_match.group(1).strip()
        total_amount = int(total_match.group(1).replace(",", ""))

        # Extract line items
        line_items = []
        for match in re.finditer(r"-\s*(.*?)\s*-\s*\$(\d+(?:,\d{3})*)", text):
            description = match.group(1).strip()
            amount = int(match.group(2).replace(",", ""))
            line_items.append({
                "description": description,
                "total": amount
            })

        return {
            "invoice_number": invoice_number,
            "vendor": vendor,
            "total_amount": total_amount,
            "line_items": line_items
        }
    except Exception as e:
        print(f"Error parsing invoice: {e}")
        return None


# -----------------------------
# Invoice Agent
# -----------------------------
class InvoiceAgent:
    """
    Invoice Processing Agent with persona-based expert consultation.
    """

    def __init__(self):
        self.persona = "You are an Invoice Processing Agent, specialized in handling invoices efficiently."
        self.goals = """
        Your goal is to process invoices accurately. For each invoice:
        1. Extract key details such as vendor, amount, and line items.
        2. Generate a one-sentence summary of the expenditure.
        3. Categorize the expenditure using an expert.
        4. Validate the invoice against purchasing policies.
        5. Store the processed invoice with categorization and validation status.
        6. Return a summary of the invoice processing results.
        """

    def process_invoice(self, invoice_text: str) -> str:
        """
        Process an invoice through the complete workflow.

        Args:
            invoice_text: Raw invoice text

        Returns:
            Summary of processing results
        """
        # Step 1: Extract invoice data
        print("Extracting invoice data...")
        invoice = extract_invoice_data(invoice_text)

        if not invoice:
            return "Error: Unable to parse invoice. Please check the format."

        # Step 2: Generate summary for categorization
        summary = f"Purchase from {invoice['vendor']} totaling ${invoice['total_amount']}"

        # Step 3: Categorize expenditure using expert
        print("Consulting categorization expert...")
        category = categorize_expenditure(summary)

        # Step 4: Validate against purchasing policies using expert
        print("Consulting compliance expert...")
        compliance = check_purchasing_rules(invoice)

        status = "Passed" if compliance.get("compliant", False) else "Failed"
        issues = compliance.get("issues", "")

        # Step 5 & 6: Build and return summary
        result = f"""Invoice #{invoice['invoice_number']}
- Vendor: {invoice['vendor']}
- Total Amount: ${invoice['total_amount']}
- Categorized as: {category}
- Compliance Check: {status}"""

        if issues:
            result += f"\n- Issues: {issues}"

        result += "\n- Stored successfully"

        return result

    def run(self, task: str) -> str:
        """
        Run the agent with a given task.

        Args:
            task: Task description (e.g., "Process this invoice: ...")

        Returns:
            Processing results
        """
        # Extract invoice text from task
        if "Process this invoice:" in task:
            invoice_text = task.split("Process this invoice:")[1].strip()
            return self.process_invoice(invoice_text)
        else:
            return "Error: Please provide a task in the format 'Process this invoice: [invoice text]'"


# -----------------------------
# Agent Factory
# -----------------------------
def create_invoice_agent() -> InvoiceAgent:
    """
    Create and configure an invoice processing agent.

    Returns:
        Configured InvoiceAgent instance
    """
    return InvoiceAgent()


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    invoice_text = """
    Invoice #4567
    Date: 2025-02-01
    Vendor: Tech Solutions Inc.
    Items: 
      - Laptop - $1,200
      - External Monitor - $300
    Total: $1,500
    """

    # Create an agent instance
    agent = create_invoice_agent()

    # Process the invoice
    print(f"Agent Persona: {agent.persona}\n")
    print(f"Agent Goals: {agent.goals}\n")
    print("=" * 50)

    response = agent.run(f"Process this invoice:\n\n{invoice_text}")

    print("\n" + "=" * 50)
    print("\nRESULT:")
    print(response)