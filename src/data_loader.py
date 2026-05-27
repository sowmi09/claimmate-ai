import pandas as pd
from .config import POLICY_CSV

def load_policies(path=POLICY_CSV):
    df = pd.read_csv(path)
    required = [
        "product_category",
        "policy_section",
        "clause_type",
        "clause_text",
        "required_documents",
        "risk_type",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in policy CSV: {missing}")
    return df

def row_to_document(row):
    return (
        f"Product Category: {row['product_category']}\n"
        f"Policy Section: {row['policy_section']}\n"
        f"Clause Type: {row['clause_type']}\n"
        f"Clause Text: {row['clause_text']}\n"
        f"Required Documents: {row['required_documents']}\n"
        f"Risk Type: {row['risk_type']}"
    )
