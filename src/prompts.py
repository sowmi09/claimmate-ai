from .config import PROMPT_DIR

def load_prompt(name: str) -> str:
    path = PROMPT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")

def format_context(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[Evidence {i}]\n"
            f"Product Category: {r.get('product_category')}\n"
            f"Policy Section: {r.get('policy_section')}\n"
            f"Clause Type: {r.get('clause_type')}\n"
            f"Clause Text: {r.get('clause_text')}\n"
            f"Required Documents: {r.get('required_documents')}\n"
            f"Score: {r.get('score'):.4f}\n"
        )
    return "\n".join(lines)
