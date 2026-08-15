#!/usr/bin/env python3
"""
AI Data Augmentor
------------------
Takes a CSV of company names and augments each row with:
    website, hq_address, phone, source_url, confidence

Design principles (see README.md for full rationale):
  - Every field must be backed by a cited source URL, or left null.
  - The model is explicitly instructed NOT to guess. A blank field is a
    correct output; a fabricated field is a failure.
  - Rate-limited, resumable: partial runs write progress incrementally
    so a 50+ row run can be safely re-started.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    pip install -r requirements.txt
    python src/augment.py --input data/starter-companies.csv --output data/augmented-companies.csv
"""

import argparse
import csv
import json
import os
import sys
import time

import anthropic

SYSTEM_PROMPT = """You are a data-augmentation agent. Given a company name, find its
official website, headquarters location, and public phone number.

Rules:
1. Prefer the company's own official website over directories or aggregators.
2. For every field you fill in, you must be able to point to the source URL
   you found it on. If you cannot verify a field from a real source, return
   null for that field. Do NOT guess, infer, or fabricate a plausible-looking
   value under any circumstances.
3. If the company name is ambiguous (multiple companies share the name),
   pick the one most consistent with the outdoor/apparel industry context,
   and note the ambiguity in the "notes" field.
4. Phone numbers: format as found on the official source; many smaller
   brands do not publish one publicly -- that is an expected, correct null.
5. Return ONLY valid JSON matching this schema, no other text:
{
  "website": "string or null",
  "hq_address": "string or null",
  "phone": "string or null",
  "source_url": "string or null",
  "confidence": "High | Medium | Low",
  "notes": "string, empty if none"
}
"""

FIELDNAMES = [
    "company_name", "website", "hq_address", "phone",
    "source_url", "confidence", "notes",
]


def augment_company(client: anthropic.Anthropic, company_name: str) -> dict:
    """Look up one company via Claude + the web_search tool, return a structured row."""
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[
            {"role": "user", "content": f"Company name: {company_name}"}
        ],
    )

    # Pull the final text block (the JSON payload) out of the response.
    text_blocks = [b.text for b in message.content if b.type == "text"]
    raw = "\n".join(text_blocks).strip()

    try:
        # Model may wrap JSON in a code fence despite instructions; strip if present.
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = {
            "website": None, "hq_address": None, "phone": None,
            "source_url": None, "confidence": "Low",
            "notes": f"PARSE_ERROR: raw model output could not be parsed as JSON: {raw[:200]}",
        }

    row = {"company_name": company_name}
    for field in FIELDNAMES[1:]:
        row[field] = data.get(field)
    return row


def main():
    parser = argparse.ArgumentParser(description="Augment a company list with website, HQ, and phone data.")
    parser.add_argument("--input", default="data/starter-companies.csv")
    parser.add_argument("--output", default="data/augmented-companies.csv")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between API calls.")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ERROR: ANTHROPIC_API_KEY is not set. Export it before running this script.")

    client = anthropic.Anthropic(api_key=api_key)

    with open(args.input, newline="") as f:
        companies = [row[0] for row in csv.reader(f)][1:]  # skip header

    # Resume support: skip companies already present in the output file.
    done = set()
    if os.path.exists(args.output):
        with open(args.output, newline="") as f:
            done = {row["company_name"] for row in csv.DictReader(f)}

    write_header = not os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for name in companies:
            if name in done:
                continue
            print(f"Looking up: {name} ...", file=sys.stderr)
            try:
                row = augment_company(client, name)
            except Exception as e:
                row = {
                    "company_name": name, "website": None, "hq_address": None,
                    "phone": None, "source_url": None, "confidence": "Low",
                    "notes": f"API_ERROR: {e}",
                }
            writer.writerow(row)
            f.flush()
            time.sleep(args.delay)

    print(f"Done. Output written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
