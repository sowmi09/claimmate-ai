import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import mlflow

from src.pipeline import ClaimMatePipeline
from src.config import (
    DATA_DIR,
    OUTPUT_DIR,
    TOP_K,
    EMBEDDING_MODEL_NAME,
    OLLAMA_MODEL,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    PROMPT_DIR,
)


def contains_expected(retrieved, expected_text):
    """
    Simple retrieval hit check.
    TRUE if any expected clause phrase appears in retrieved section/clause text.
    """
    expected_parts = [p.strip().lower() for p in str(expected_text).split(";")]

    retrieved_text = " ".join(
        f"{r.get('policy_section', '')} {r.get('clause_text', '')}"
        for r in retrieved
    ).lower()

    return any(part in retrieved_text for part in expected_parts)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    eval_path = DATA_DIR / "sample_claims.csv"
    eval_df = pd.read_csv(eval_path)

    pipeline = ClaimMatePipeline()

    evaluation_rows = []
    evidence_rows = []

    hits = 0

    for idx, row in eval_df.iterrows():
        query = row["query"]
        expected_clause = row["expected_clause"]

        result = pipeline.run(query, top_k=TOP_K)
        retrieved = result["retrieved"]

        hit = contains_expected(retrieved, expected_clause)
        hits += int(hit)

        evaluation_rows.append(
            {
                "query_id": idx + 1,
                "query": query,
                "expected_clause": expected_clause,
                "hit": hit,
                "route": result["route"],
                "answer_preview": result["answer"][:300],
            }
        )

        for rank, item in enumerate(retrieved, start=1):
            evidence_rows.append(
                {
                    "query_id": idx + 1,
                    "query": query,
                    "rank": rank,
                    "score": item.get("score"),
                    "product_category": item.get("product_category"),
                    "policy_section": item.get("policy_section"),
                    "clause_type": item.get("clause_type"),
                    "clause_text": item.get("clause_text"),
                    "required_documents": item.get("required_documents"),
                    "source_file": item.get("source_file", ""),
                    "chunk_id": item.get("chunk_id", ""),
                }
            )

    total_questions = len(eval_df)
    retrieval_hit_rate = hits / total_questions if total_questions else 0

    evaluation_results_path = OUTPUT_DIR / "evaluation_results.csv"
    retrieved_evidence_path = OUTPUT_DIR / "retrieved_evidence.csv"
    summary_path = OUTPUT_DIR / "evaluation_summary.json"
    report_path = OUTPUT_DIR / "evaluation_report.txt"

    pd.DataFrame(evaluation_rows).to_csv(evaluation_results_path, index=False)
    pd.DataFrame(evidence_rows).to_csv(retrieved_evidence_path, index=False)

    summary = {
        "run_time": datetime.now().isoformat(),
        "total_questions": total_questions,
        "hits": hits,
        "retrieval_hit_rate": retrieval_hit_rate,
        "top_k": TOP_K,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "ollama_model": OLLAMA_MODEL,
        "faiss_index_path": str(FAISS_INDEX_PATH),
        "metadata_path": str(METADATA_PATH),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_text = f"""
ClaimMate AI Evaluation Report
==============================

Total questions: {total_questions}
Hits: {hits}
Retrieval hit rate: {retrieval_hit_rate:.2f}

Embedding model: {EMBEDDING_MODEL_NAME}
Ollama model: {OLLAMA_MODEL}
Top-k: {TOP_K}

Meaning:
- hit=True means expected policy clause was found in retrieved evidence.
- hit=False may mean retrieval missed it OR the expected_clause text is stricter than the actual section name.
"""

    report_path.write_text(report_text.strip(), encoding="utf-8")

    mlflow.set_experiment("claimmate-ai-retrieval")

    with mlflow.start_run(run_name="claimmate_retrieval_eval"):
        # Params: configuration values
        mlflow.log_params(
            {
                "embedding_model": EMBEDDING_MODEL_NAME,
                "ollama_model": OLLAMA_MODEL,
                "top_k": TOP_K,
                "eval_dataset": str(eval_path),
                "faiss_index_path": str(FAISS_INDEX_PATH),
                "metadata_path": str(METADATA_PATH),
            }
        )

        # Metrics: numeric values only
        mlflow.log_metrics(
            {
                "retrieval_hit_rate": retrieval_hit_rate,
                "total_questions": total_questions,
                "hits": hits,
                "misses": total_questions - hits,
            }
        )

        # Tags: searchable metadata
        mlflow.set_tags(
            {
                "project": "ClaimMate AI",
                "stage": "retrieval_evaluation",
                "rag_type": "local_faiss_rag",
                "llm_runtime": "ollama",
            }
        )

        # Artifacts: CSVs, JSON, text report, prompts
        mlflow.log_artifact(str(evaluation_results_path), artifact_path="evaluation")
        mlflow.log_artifact(str(retrieved_evidence_path), artifact_path="evaluation")
        mlflow.log_artifact(str(summary_path), artifact_path="evaluation")
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")

        if PROMPT_DIR.exists():
            mlflow.log_artifacts(str(PROMPT_DIR), artifact_path="prompts")

    print("Evaluation completed.")
    print(f"Retrieval hit rate: {retrieval_hit_rate:.2f}")
    print("Logged to MLflow experiment: claimmate-ai-retrieval")
    print(f"Saved: {evaluation_results_path}")
    print(f"Saved: {retrieved_evidence_path}")


if __name__ == "__main__":
    main()