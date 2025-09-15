import pandas as pd
#input_path="data/Input/transactions_sample.csv"

expected_cols = [
  "transaction_id","account_id","transaction_date","amount","merchant",
  "city","country","channel","historical_avg","historical_frequency","home_country"
]

def schema_validation(input_path):
  df = pd.read_csv(input_path, parse_dates=["transaction_date"])
  missing = [c for c in expected_cols if c not in df.columns]
  print("----------Schema Validation Results:----------")
  if missing:
    print("Missing columns: required columns are:", missing)
  else:
    print("Schema validation passed. All expected columns are present.")
    print("Rows loaded:", len(df))

if __name__ == "__main__":
    schema_validation(input_path)
