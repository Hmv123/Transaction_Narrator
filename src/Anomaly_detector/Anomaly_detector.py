import pandas as pd
import numpy as np
from datetime import timedelta
# input_path="data/Input/transactions_sample.csv"
# output_path="data/Output/anomalies_output.csv"

# ---------- RULE FUNCTIONS ----------

def detect_high_value(df):
    anomalies = []
    for row in df.itertuples():
        if (row.amount > 5000) or (row.amount > 5 * row.historical_avg):
            anomalies.append((row.transaction_id, row.account_id, "HIGH_VALUE"))
    return anomalies


def detect_velocity(df):
    anomalies = []
    df_sorted = df.sort_values(by=["account_id", "transaction_date"])
    
    for acct_id, group in df_sorted.groupby("account_id"):
        times = group["transaction_date"].tolist() #converting to a list for comparison
        txn_ids = group["transaction_id"].tolist()
        
        for i in range(len(times)):
            window_start = times[i]
            window_txns = [txn_ids[i]]
            
            # look ahead transactions within 10 minutes
            for j in range(i+1, len(times)):
                if times[j] - window_start <= timedelta(minutes=10): #ref:timedelta(days=1, hours=2, minutes=30)
                    window_txns.append(txn_ids[j])
                else:
                    break
            
            if len(window_txns) > 3:
                for tid in window_txns:
                    anomalies.append((tid, acct_id, "VELOCITY"))
    return anomalies


def detect_foreign(df):
    anomalies = []
    for  row in df.itertuples():
        if row.country != row.home_country:
            anomalies.append((row.transaction_id, row.account_id, "FOREIGN"))
    return anomalies


def detect_odd_hours(df):
    anomalies = []
    for row in df.itertuples():
        hour = row.transaction_date.hour
        if 0 <= hour < 5:  # midnight to 5 AM
            anomalies.append((row.transaction_id, row.account_id, "ODD_HOURS"))
    return anomalies

# ---------- MAIN FUNCTION ----------

def detect_anomalies(input_path, output_path):
    df = pd.read_csv(input_path, parse_dates=["transaction_date"])
    
    all_anomalies = []
    
    all_anomalies.extend(detect_high_value(df))
    all_anomalies.extend(detect_velocity(df))
    all_anomalies.extend(detect_foreign(df))
    all_anomalies.extend(detect_odd_hours(df))
    
    # Remove duplicates (same anomaly_type reported multiple times)
    anomalies_df = pd.DataFrame(all_anomalies, columns=["transaction_id", "account_id", "anomaly_type"])
    
    # Map transaction_id -> list of reasons
    anomaly_map = anomalies_df.groupby("transaction_id")["anomaly_type"].apply(list).to_dict()
    # Build final tagged dataframe
    tagged_rows = []
    for  row in df.itertuples():
        tid = row.transaction_id
        if tid in anomaly_map:
            tagged_rows.append({
                "transaction_id": tid,
                "account_id": row.account_id,
                "transaction_date":row.transaction_date,
                "amount":row.amount,
                "merchant":row.merchant,
                "city":row.city,
                "country":row.country,
                "channel":row.channel,
                "historical_avg":row.historical_avg,
                "historical_frequency":row.historical_frequency,
                "home_country":row.home_country,
                "tag": "Anomalous",
                "reason_codes": ";".join(sorted(set(anomaly_map[tid])))
            })
        else:
            tagged_rows.append({
                "transaction_id": tid,
                "account_id": row.account_id,
                "transaction_date":row.transaction_date,
                "amount":row.amount,
                "merchant":row.merchant,
                "city":row.city,
                "country":row.country,
                "channel":row.channel,
                "historical_avg":row.historical_avg,
                "historical_frequency":row.historical_frequency,
                "home_country":row.home_country,
                "tag": "Normal",
                "reason_codes": "Non Suspecious"
            })
    
    
    result_df = pd.DataFrame(tagged_rows)
    result_df.to_csv(output_path, index=False)
    print("---------Anomaly Detection Results:---------")
    print(f"Transactions tagged and saved to {output_path}")
    print("Rows loaded:", len(df))
    return result_df

if __name__ == "__main__":
    detect_anomalies(input_path,output_path)