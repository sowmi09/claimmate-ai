def route_query(query: str) -> str:
    """
    Lightweight deterministic router.
    Later, this can be replaced by a local LLM router prompt.
    """
    q = query.lower()

    fraud_terms = ["fake invoice", "false claim", "fake bill", "forge", "forged", "lie to", "mislead"]
    if any(t in q for t in fraud_terms):
        return "fake_claim_or_fraud"

    email_terms = ["draft", "email", "mail", "write to", "complaint", "follow-up", "follow up"]
    if any(t in q for t in email_terms):
        return "complaint_email"

    return_terms = ["return", "refund", "replacement", "delivered", "received damaged"]
    if any(t in q for t in return_terms):
        return "return_refund"

    reject_terms = ["rejected", "denied", "not accepted", "physical damage"]
    if any(t in q for t in reject_terms):
        return "rejection_reason"

    doc_terms = ["document", "documents", "invoice", "serial number", "proof", "attach"]
    if any(t in q for t in doc_terms):
        return "document_checklist"

    service_terms = ["service", "repair", "delay", "technician", "inspection"]
    if any(t in q for t in service_terms):
        return "repair_service"

    warranty_terms = ["warranty", "covered", "claim", "eligible", "under warranty"]
    if any(t in q for t in warranty_terms):
        return "warranty_eligibility"

    return "warranty_eligibility"
