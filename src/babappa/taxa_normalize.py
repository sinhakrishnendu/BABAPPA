from __future__ import annotations


def normalize_taxon(name: str) -> str:
    text = str(name).strip()
    if "|" in text:
        text = text.split("|", 1)[0].strip()
    if "/" in text:
        text = text.split("/", 1)[0].strip()
    text = "_".join(text.split())
    return text.lower()
