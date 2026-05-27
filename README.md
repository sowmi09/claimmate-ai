# ClaimMate AI

**ClaimMate AI** is a local RAG-based warranty, return, refund, and service-claim assistant.

It helps users understand product warranty policies, check claim eligibility, identify missing documents, explain claim rejection reasons, and draft polite customer support emails.

This project is built as a GitHub portfolio project for GenAI + RAG + MLOps.

---

## What the user can ask

```text
My laptop stopped working after 8 months. Is it covered under warranty?

The service center rejected my claim saying physical damage. What should I check?

What documents do I need for a warranty claim?

Draft a polite email asking for warranty claim review.

I received a damaged product yesterday. Can I return it?
```

---

## Core idea

```text
User question
    ↓
Prompt router
    ↓
FAISS retrieves relevant warranty/return policy clauses
    ↓
Local LLM answers using only retrieved evidence
    ↓
Bot returns:
    - claim status
    - reason
    - missing documents
    - next action
    - optional email draft
```

---

## Tech stack

| Area | Tool |
|---|---|
| UI | Streamlit |
| Retrieval | FAISS |
| Embeddings | Hugging Face Transformers directly |
| Local LLM | Ollama |
| Prompting | Separate prompt files |
| MLOps | MLflow-ready logging |
| Versioning | Git + optional DVC |
| Containerization | Docker skeleton |
| CI | GitHub Actions |

This project intentionally avoids the repeated `sentence-transformers/all-MiniLM-L6-v2 + ChromaDB + OpenAI API` pattern.

---

## Project structure

```text
claimmate-ai/
├── app.py
├── build_index.py
├── evaluate.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/
│   ├── policies.csv
│   ├── sample_claims.csv
│   └── qlora_sft_claimmate.jsonl
├── prompts/
│   ├── router_prompt.txt
│   ├── evidence_prompt.txt
│   ├── claim_decision_prompt.txt
│   └── email_prompt.txt
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── embeddings_hf.py
│   ├── faiss_store.py
│   ├── prompts.py
│   ├── llm_ollama.py
│   ├── claim_router.py
│   ├── claim_analyzer.py
│   └── pipeline.py
├── finetuning/
│   ├── README.md
│   └── prepare_sft_dataset.py
├── tests/
│   └── test_router.py
└── .github/
    └── workflows/
        └── ci.yml
```

---

## Setup

### 1. Create environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Optional: install and run Ollama

Install Ollama and pull a small local model:

```bash
ollama pull qwen2.5:3b
```

You can change the model name in `src/config.py`.

If Ollama is not running, the app still works with a simple fallback response generator.

### 4. Build FAISS index

```bash
python build_index.py
```

### 5. Run app

```bash
streamlit run app.py
```

---

## Example output

User:

```text
My laptop stopped working after 8 months. I have invoice and serial number. The service center rejected it saying physical damage, but there is no visible damage. Draft an email.
```

Bot returns:

```text
Claim status: Needs review / unclear

Reason:
- The product may still be inside the 12-month warranty period.
- The policy excludes physical damage.
- Since rejection is based on physical damage, request written inspection details.

Missing / useful documents:
- Invoice
- Serial number
- Product photos
- Service report or rejection note

Suggested next action:
Ask the service center to share inspection evidence and request a warranty review.

Email draft:
Subject: Request for Warranty Claim Review

Dear Customer Support Team,
...
```

---

## MLOps plan

This repo includes a simple `evaluate.py` script that can log retrieval quality to MLflow if MLflow is installed.

Track:

- embedding model
- FAISS index version
- prompt version
- top-k
- retrieval hit rate
- output file path

---

## QLoRA / LoRA plan

Version 1 does not require fine-tuning.

For Version 2, we can use QLoRA to teach the LLM structured behavior:

- claim eligibility reasoning
- missing document detection
- polite email drafting
- refusal when user asks legal guarantee or fake claim wording

The dataset starter is available at:

```text
data/qlora_sft_claimmate.jsonl
```

---

## Limitations

- This is not legal advice.
- The bot should not guarantee warranty approval.
- It answers from the provided policy dataset.
- Real brand policies must be used carefully and kept updated.

---

## Resume line

Built **ClaimMate AI**, a local RAG-based warranty claim assistant using FAISS, Hugging Face embeddings, Ollama local LLM prompting, and MLOps-ready evaluation to analyze policy clauses, identify missing documents, explain rejection reasons, and draft customer support emails.


Version 1 supports:
- Structured CSV warranty policy data
- Synthetic PDF warranty/return/refund policy documents
- PDF text extraction using pypdf
- Chunking and FAISS indexing
- Local LLM answer generation with Ollama