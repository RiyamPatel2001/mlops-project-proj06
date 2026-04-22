import os
import json
from collections import Counter

import requests

_API_URL = "https://api.anthropic.com/v1/messages"
_MODEL = "claude-sonnet-4-20250514"
_MAX_TOKENS = 100


def name_cluster(payees: list[str], existing_labels: list[str]) -> str:
    """
    Ask the Anthropic API to suggest a short category name for a cluster.

    Falls back to the most common existing_label on any API failure.
    """
    payee_block = "\n".join(f"- {p}" for p in payees)
    label_block = "\n".join(f"- {l}" for l in existing_labels) if existing_labels else "(none)"

    prompt = (
        "You are a financial transaction categorizer.\n\n"
        f"Payees in this cluster:\n{payee_block}\n\n"
        f"Existing category labels:\n{label_block}\n\n"
        "Suggest a single short category name (2-4 words) that best describes "
        "this group of transactions. Respond with only the category name, nothing else."
    )

    headers = {"Content-Type": "application/json", "anthropic-version": "2023-06-01"}
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    payload = {
        "model": _MODEL,
        "max_tokens": _MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        resp = requests.post(_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"].strip()
        return text.split("\n")[0].strip()
    except Exception:
        if existing_labels:
            return Counter(existing_labels).most_common(1)[0][0]
        return "Uncategorized"
