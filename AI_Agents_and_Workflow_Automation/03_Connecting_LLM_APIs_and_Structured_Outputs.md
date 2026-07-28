# 03: Connecting LLM APIs and Enforcing Structured Outputs

This module explores **LLM API Integration in Automation Workflows**, covering multi-provider API connections (OpenAI, Anthropic, Gemini, Ollama), dynamic variable prompt injection, schema enforcement (JSON Schema & Pydantic), automated retry mechanisms, and hands-on implementations in n8n and Python.

---

## 1. LLM API Ingestion Architecture in Workflow Engines

Integrating Large Language Models (LLMs) into automated workflows requires moving beyond basic text generation to **deterministic data extraction and transformation**. Workflow nodes pass structured JSON payloads into model context windows and parse model outputs back into valid workflow variables.

---

## 2. Structured Outputs Calculus: Temperature & Schema Enforcement

To ensure downstream automation nodes (e.g., PostgreSQL, Jira, or Slack) receive predictable schema inputs without runtime execution failures, workflow engines must enforce strict structural constraints.

### 1. Sampling Temperature Selection Matrix
*   **Deterministic Extraction ($\text{Temperature} = 0.0$):** Eliminates non-deterministic sampling variance. Essential for JSON extraction, classification, entity recognition, and decision routing.
*   **Balanced Logic ($\text{Temperature} = 0.2 - 0.3$):** Standard for summarized support emails, customer sentiment scoring, and natural language query generation.
*   **Creative Generation ($\text{Temperature} \ge 0.7$):** Used for marketing copy or personalized outreach email drafting.

---

### 2. Schema Validation Retry Probability
When relying on prompt-based JSON outputs without native Function Calling / Structured Outputs, the probability of receiving a valid JSON payload on attempt $k$ decreases as schema complexity increases.

Let $P_{valid}$ be the single-attempt probability of a valid JSON output. The cumulative success probability $P_{success}(N)$ over $N$ retries with self-correction feedback is:

$$P_{success}(N) = 1 - (1 - P_{valid})^N$$

*   **Native Structured Outputs (OpenAI `response_format={"type": "json_schema"}` / Anthropic Tool Calling):** Guarantees $P_{valid} \approx 1.0$ by constraining model token generation at the decoding layer.

---

## 3. Error Handling Patterns: Exponential Backoff & Fallback Models

When relying on external AI providers, workflow engines must implement multi-tier resilience logic:

1.  **Rate-Limit Retries (HTTP 429 / 503):** Exponential backoff with randomized jitter.
2.  **Schema Mismatch Recovery:** Feeding the validation error log back into the LLM context prompt to generate a corrected schema.
3.  **Model Fallback Cascades:** Primary Model (e.g., `gpt-4o`) $\rightarrow$ Secondary Model (e.g., `claude-3-5-sonnet`) $\rightarrow$ On-Premise Local Model (`ollama/llama3.1`).

---

## 4. Production Implementation: Multi-Provider LLM & Schema Engine in Python

Here is a complete, production-grade Python script using **Pydantic** and **FastAPI** that acts as an LLM Automation Node, processing unstructured incoming payloads into strictly validated JSON outputs with automatic retries.

```python
import os
import json
import logging
from typing import Optional, List
from enum import Enum
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, EmailStr
from openai import OpenAI

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("LLMAutomationEngine")

app = FastAPI(
    title="LLM Structured Output Automation Node",
    description="Production API wrapper converting raw text into validated JSON schema for n8n/Make workflows.",
    version="1.0.0"
)

# Initialize OpenAI Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "mock-key-for-development"))


# -------------------------------------------------------------------
# 1. Pydantic Target Schema Definitions
# -------------------------------------------------------------------
class PriorityEnum(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    URGENT = "URGENT"


class CategoryEnum(str, Enum):
    BILLING = "BILLING"
    TECHNICAL_BUG = "TECHNICAL_BUG"
    FEATURE_REQUEST = "FEATURE_REQUEST"
    ACCOUNT_ACCESS = "ACCOUNT_ACCESS"


class TicketExtractionSchema(BaseModel):
    """Enforced schema structure for automated customer support ingestion."""
    customer_name: Optional[str] = Field(description="Extracted full name of the sender.")
    customer_email: Optional[EmailStr] = Field(description="Valid email address if present in text.")
    category: CategoryEnum = Field(description="Primary category classification of the issue.")
    priority: PriorityEnum = Field(description="Assessed priority level based on customer urgency and sentiment.")
    summary: str = Field(description="A concise 1-sentence summary of the request.")
    action_items: List[str] = Field(description="List of specific technical action items required.")


class IngestionRequestPayload(BaseModel):
    """Incoming request payload from n8n / Zapier webhook."""
    raw_text: str = Field(..., example="Hi, my name is Neeraj. I'm locked out of my account and my production pipeline is down! Fix this ASAP!")


# -------------------------------------------------------------------
# 2. LLM Extraction Processing Node
# -------------------------------------------------------------------
def extract_structured_ticket(raw_input: str, max_retries: int = 2) -> TicketExtractionSchema:
    """Invokes OpenAI Chat Completion using native Structured Outputs (JSON Schema constraint)."""
    
    system_prompt = (
        "You are an enterprise support ticket classification node. "
        "Analyze the incoming unstructured message and extract structural metadata strictly conforming to the requested schema."
    )

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"🤖 Invoking LLM Extraction Engine (Attempt {attempt}/{max_retries})...")
            
            # Utilizing beta parse for structured Pydantic enforcement
            response = client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Incoming Customer Message:\n{raw_input}"}
                ],
                response_format=TicketExtractionSchema,
                temperature=0.0
            )

            extracted_data = response.choices[0].message.parsed
            logger.info("✅ Successfully extracted and validated structured output.")
            return extracted_data

        except Exception as e:
            logger.warning(f"⚠️ Extraction Attempt {attempt} failed: {str(e)}")
            if attempt == max_retries:
                logger.error("❌ Exceeded maximum retries for structured extraction.")
                raise e


# -------------------------------------------------------------------
# 3. API Endpoint Route
# -------------------------------------------------------------------
@app.post("/api/v1/automation/extract-ticket", response_model=TicketExtractionSchema, status_code=status.HTTP_200_OK)
async def process_llm_ticket_extraction(payload: IngestionRequestPayload):
    """
    HTTP POST Endpoint for n8n/Make HTTP Request Nodes.
    Accepts raw text, runs LLM parsing with Pydantic validation, returns clean JSON payload.
    """
    if not payload.raw_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Raw input text cannot be empty."
        )

    try:
        result = extract_structured_ticket(payload.raw_text)
        return result
    except Exception as err:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to extract structured data from LLM: {str(err)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("\n--- Starting LLM Automation Parsing Service on Port 8000 ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
