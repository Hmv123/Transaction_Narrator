import os
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
# input_path="data/Output/anomalies_output.csv"
# output_path="data/Output/anomalies_with_narratives.csv"

# Load Azure credentials
load_dotenv()
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION")
)
chat_model=os.getenv('AZURE_OPENAI_CHAT_DEPLOYMENT')


# ---------- PROMPT TEMPLATE ----------
def build_prompt(row):
    example_narrative = (
        "Example: Transaction T12345 was flagged as HIGH_VALUE because the amount "
        "of $12,000 exceeded 5 times the historical average of $2000."
    )
    
    return f"""
You are a fraud detection assistant.
Explain why this transaction was flagged, using ONLY the provided data.Don't hallucinate.
Follow the style of the example narrative below.

{example_narrative}

Transaction details:
- ID: {row.transaction_id}
- Account: {row.account_id}
- Amount: {getattr(row, 'amount', 'N/A')}
- Merchant: {getattr(row, 'merchant', 'N/A')}
- City: {getattr(row, 'city', 'N/A')}
- Country: {getattr(row, 'country', 'N/A')}
- Channel: {getattr(row, 'channel', 'N/A')}
- Historical Average: {getattr(row, 'historical_avg', 'N/A')}
- Historical Frequency: {getattr(row, 'historical_frequency', 'N/A')}
- Home Country: {getattr(row, 'home_country', 'N/A')}

Reason Codes: {row.reason_codes}

Rules:
- HIGH_VALUE: amount > 5000 OR amount > 5 × historical average
- VELOCITY: >3 transactions within 10 minutes
- FOREIGN: country ≠ home_country
- ODD_HOURS: time between 00:00–05:00

Task:
Write a short narrative (2–3 sentences) explaining why this transaction was flagged.
Use clear and factual language. Do not add details not present in the data.
"""


def generate_narratives(input_path, output_path):
    df = pd.read_csv(input_path, parse_dates=["transaction_date"])

    narratives = []
    for row in df.itertuples():
        if row.tag == "Anomalous":
            prompt = build_prompt(row)

            response = client.chat.completions.create(
                model=chat_model,  # Replace with your Azure deployment name
                messages=[
                    {"role": "system", "content": "You are a precise fraud detection assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Keep outputs factual and stable
                max_tokens=200
            )

            narrative = response.choices[0].message.content.strip()
        else:
            narrative = "This transaction qualifies the normal transaction criteria. Hence Normal tag is applied."

        narratives.append(narrative)

    df["narrative"] = narratives
    outdf=df[["transaction_id","account_id","tag","reason_codes","narrative"]]
    outdf.to_csv(output_path, index=False)
    print(f"Narratives generated and saved to {output_path}")


if __name__ == "__main__":
    generate_narratives(input_path, output_path)
