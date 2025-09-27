# ------------------- IMPORTS -------------------
import os
import json
import pandas as pd
import sqlite3
from dotenv import load_dotenv
from openai import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ------------------- ENV SETUP -------------------
load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION")
)

chat_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

embedder = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION"),
    model="text-embedding_3_large",
    chunk_size=1000
)

# ------------------- SQLITE SETUP -------------------

def init_embedding_db(db_path="data/vector_embeddings.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS embeddings")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            transaction_id TEXT PRIMARY KEY,
            account_id TEXT,
            tag TEXT,
            reason_codes TEXT,
            narrative TEXT,
            embedding TEXT,
            source TEXT,
            amount REAL,
            embedding_text TEXT
        )
    """)
    conn.commit()
    return conn

# ------------------- PROMPT BUILDER -------------------

def build_prompt(row):
    example = (
        "Example: Transaction T12345 was flagged as HIGH_VALUE because the amount "
        "of $12,000 exceeded 5 times the historical average of $2000."
    )
    return f"""
You are a fraud detection assistant.
Explain why this transaction was flagged, using ONLY the provided data. Don't hallucinate.
Follow the style of the example narrative below.

{example}

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

# ------------------- COSINE SIMILARITY SEARCH -------------------

def search_similar_narratives(query, db_path="data/vector_embeddings.db", top_k=5):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query_vec = embedder.embed_query(query)

    cursor.execute("SELECT transaction_id, narrative, embedding FROM embeddings")
    results = []

    for txn_id, narrative, vec_json in cursor.fetchall():
        vec = json.loads(vec_json)
        score = cosine_similarity([query_vec], [vec])[0][0]
        results.append((txn_id, narrative, score))

    conn.close()
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_k]

# ------------------- MAIN FUNCTION -------------------

def generate_narratives(input_path, output_path, db_path="data/vector_embeddings.db"):
    conn = init_embedding_db(db_path)
    cursor = conn.cursor()
    output_rows = []

    chunksize = 1000
    reader = pd.read_csv(input_path, chunksize=chunksize)
    for chunk in reader:
        chunk["transaction_date"] = pd.to_datetime(chunk["transaction_date"], errors="coerce", format="%d-%m-%Y %H:%M")    
        for row in chunk.itertuples():
            cursor.execute("SELECT narrative FROM embeddings WHERE transaction_id = ?", (row.transaction_id,))
            result = cursor.fetchone()

            if result:
                narrative = result[0]
                source = "cached"
            else:
                if row.tag == "Anomalous":
                    prompt = build_prompt(row)
                    response = client.chat.completions.create(
                        model=chat_model,
                        messages=[
                            {"role": "system", "content": "You are a precise fraud detection assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2,
                        max_tokens=200
                    )
                    narrative = response.choices[0].message.content.strip()
                else:
                    narrative = "This transaction qualifies the normal transaction criteria. Hence Normal tag is applied."

                # Build metadata-rich embedding text
                embedding_text = (
                    f"Transaction ID: {row.transaction_id} | "
                    f"Account ID: {row.account_id} | "
                    f"Amount: ${row.amount:.2f} | "
                    f"Tag: {row.tag} | "
                    f"Reason Codes: {row.reason_codes} | "
                    f"Narrative: {narrative}"
                )

                embedding = embedder.embed_query(embedding_text)
                embedding_json = json.dumps(embedding)
                source = "generated"

                cursor.execute("""
                    INSERT INTO embeddings (
                        transaction_id, account_id, tag, reason_codes, narrative, embedding, source, amount, embedding_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row.transaction_id,
                    row.account_id,
                    row.tag,
                    row.reason_codes,
                    narrative,
                    embedding_json,
                    source,
                    row.amount,
                    embedding_text
                ))

            output_rows.append({
                "transaction_id": row.transaction_id,
                "account_id": row.account_id,
                "tag": row.tag,
                "reason_codes": row.reason_codes,
                "narrative": narrative,
                "source": source
            })

        conn.commit()

    conn.close()

    outdf = pd.DataFrame(output_rows)
    outdf.to_csv(output_path, index=False)

    print(f"✅ Narratives and embeddings stored in SQLite: {db_path}")
    print(f"📄 Final output saved to CSV: {output_path}")

# ------------------- EXECUTION -------------------

# input_path = "data/Output/anomalies_output.csv"
# output_path = "data/Output/anomalies_with_narratives.csv"
# generate_narratives(input_path, output_path)
# query = "Why was this flagged as HIGH_VALUE?"

if __name__ == "__main__":
    generate_narratives(input_path, output_path)
    search_similar_narratives(query)
