from .prompts import load_prompt, format_context
from .llm_ollama import call_ollama, ollama_available

def fallback_answer(query: str, route: str, retrieved: list[dict]) -> str:
    """
    Fallback response when Ollama is not running.
    This keeps the app usable for demo and GitHub reviewers.
    """
    if route == "fake_claim_or_fraud":
        return (
            "I cannot help create fake invoices, false claims, or misleading warranty descriptions.\n\n"
            "I can help you write a truthful customer support email using genuine facts and documents."
        )

    context_lines = []
    docs = set()
    for r in retrieved:
        context_lines.append(f"- {r['policy_section']}: {r['clause_text']}")
        if r.get("required_documents"):
            for d in str(r["required_documents"]).split(","):
                docs.add(d.strip())

    status = "unclear / needs review"
    q = query.lower()
    if "8 months" in q or "6 months" in q or "within warranty" in q:
        status = "possibly eligible, subject to inspection"
    if "cracked" in q or "liquid" in q or "physical damage" in q:
        status = "needs review or may not be eligible if the exclusion is confirmed"

    email_block = ""
    if route == "complaint_email":
        email_block = (
            "\n\nEmail draft:\n\n"
            "Subject: Request for Warranty Claim Review\n\n"
            "Dear Customer Support Team,\n\n"
            "I am requesting a review of my warranty/service claim. Based on the available policy information, "
            "I would like to understand the reason for the decision and the next steps required from my side.\n\n"
            "Please share the written inspection details or supporting reason for the claim decision. "
            "I am ready to provide the required documents such as invoice, serial number, product photos, "
            "service request number, or service report wherever applicable.\n\n"
            "Kindly review my case and guide me on the next steps.\n\n"
            "Thank you.\n"
        )

    doc_lines = "\n".join(f"- {d}" for d in sorted(docs) if d and d != "none") or "- Not clearly specified"

    return (
        f"Claim status: {status}\n\n"
        f"Relevant policy evidence:\n" + "\n".join(context_lines[:4]) + "\n\n"
        f"Missing or useful documents:\n{doc_lines}\n\n"
        "Suggested next action:\n"
        "- Collect the required documents and ask the service/support team for written inspection details or next steps."
        + email_block
    )

def analyze_claim(query: str, route: str, retrieved: list[dict]) -> str:
    if route == "fake_claim_or_fraud":
        return fallback_answer(query, route, retrieved)

    context = format_context(retrieved)

    if route == "complaint_email":
        system_prompt = load_prompt("email_prompt.txt")
    else:
        system_prompt = load_prompt("claim_decision_prompt.txt")

    user_prompt = (
        f"User query:\n{query}\n\n"
        f"Retrieved policy context:\n{context}"
    )

    if ollama_available():
        try:
            return call_ollama(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:
            return (
                "Ollama was detected but the call failed, so fallback output is shown.\n"
                f"Error: {exc}\n\n"
                + fallback_answer(query, route, retrieved)
            )

    return (
        "Ollama is not running, so fallback output is shown. "
        "Start Ollama for local LLM-generated answers.\n\n"
        + fallback_answer(query, route, retrieved)
    )
