from typing import Any, Dict


def deterministic_programmatic(extraction: Dict[str, Any], truth: Dict[str, Any]) -> float:
    """Deterministic 0-1 grader used by the validator."""
    extraction = extraction or {}

    def clean(v: Any) -> str:
        return str(v or "").strip().lower()

    qty_truth = clean(truth.get("qty"))
    qty_guess = clean(extraction.get("qty"))
    qty_score = 1.0 if qty_guess and (qty_guess in qty_truth or qty_truth in qty_guess) else 0.0

    extraction_score = (
        qty_score
        + (1.0 if clean(extraction.get("category")) == clean(truth.get("category")) else 0.0)
        + (1.0 if clean(extraction.get("Destination")) == clean(truth.get("Destination")) else 0.0)
        + (1.0 if clean(extraction.get("Origin")) == clean(truth.get("Origin")) else 0.0)
    ) / 4.0

    selected_laws = set(extraction.get("laws", []))
    required_laws = set(truth.get("all_required_laws", []))
    if not required_laws:
        required_laws = set(truth.get("required_export_laws", [])) | set(truth.get("required_import_laws", []))
    law_match_score = len(selected_laws & required_laws) / max(1, len(required_laws))
    law_extras = selected_laws - required_laws
    red_herrings = set(truth.get("red_herrings", []))
    law_penalty = sum(0.5 if l in red_herrings else 1.0 for l in law_extras) / max(1, len(required_laws))
    law_score = max(0.0, law_match_score - law_penalty)

    regulator_targets = [clean(r) for r in [truth.get("origin_regulator"), truth.get("dest_regulator")] if r and r != "N/A"]
    regulator_guess = clean(extraction.get("regulator"))
    regulator_score = sum(1 for r in regulator_targets if r and r in regulator_guess) / max(1, len(regulator_targets))

    required_docs = truth.get("import_rules", {}).get("documents", []) + truth.get("export_rules", {}).get("documents", [])
    matched_docs = set()
    for doc in extraction.get("documents", []):
        clean_doc = clean(doc)
        for req_doc in required_docs:
            if clean_doc in clean(req_doc) or clean(req_doc) in clean_doc:
                matched_docs.add(req_doc)
    document_score = len(matched_docs) / max(1, len(required_docs))

    final_score = (
        0.35 * extraction_score
        + 0.35 * law_score
        + 0.15 * regulator_score
        + 0.15 * document_score
    )
    return round(max(0.01, min(0.99, final_score)), 2)